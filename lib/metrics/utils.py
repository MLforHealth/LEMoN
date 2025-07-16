import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.optimize import bisect, minimize_scalar
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, 
                             brier_score_loss, balanced_accuracy_score, recall_score, classification_report, confusion_matrix)
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from scipy.optimize import fminbound
from scipy.optimize import minimize
from lib.datasets.utils import cifar10_labels, cifar100_labels
from itertools import product
from torch.optim import LBFGS
from netcal.metrics import ECE

def combinations_base(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))

def calc_scores_given_hparams(df, best_hparams, return_dn = False):
    d_ns, d_ms = [], []
    for idx, row in df.iterrows(): # inefficient
        scaling_factor = np.ones(len(row['D_n']))
        scaling_factor *= np.exp(-best_hparams['tau_1_n']*(row['D_n']))
        scaling_factor *= np.exp(-best_hparams['tau_2_n']*row['dists_tr_n'])
        d_n = np.dot(scaling_factor.flatten(), row['dists_n'].flatten())/len(row['D_n'])

        scaling_factor = np.ones(len(row['D_m']))
        scaling_factor *= np.exp(-best_hparams['tau_1_m']*(row['D_m']))
        scaling_factor *= np.exp(-best_hparams['tau_2_m']*row['dists_tr_m'])
        d_m = np.dot(scaling_factor.flatten(), row['dists_m'].flatten())/len(row['D_m'])

        d_ns.append(d_n)
        d_ms.append(d_m)

    d_ns = np.array(d_ns)
    d_ms = np.array(d_ms)

    scores = df['d_1'] + best_hparams['beta'] * d_ns + best_hparams['gamma'] * d_ms

    if return_dn:
        return scores, d_ns, d_ms
    else:
        return scores  

def calc_scores_given_hparams_vectorized(df, best_hparams, return_dn=False, torch_arr=False):
    if torch_arr:
        D_ns = torch.stack([torch.tensor(d) for d in df['D_n'].values])
        D_ms = torch.stack([torch.tensor(d) for d in df['D_m'].values])
        dists_tr_ns = torch.stack([torch.tensor(d) for d in df['dists_tr_n'].values])
        dists_tr_ms = torch.stack([torch.tensor(d) for d in df['dists_tr_m'].values])
        dists_ns = torch.stack([torch.tensor(d) for d in df['dists_n'].values])
        dists_ms = torch.stack([torch.tensor(d) for d in df['dists_m'].values])

        scaling_factors_n = torch.exp(-best_hparams['tau_1_n'] * D_ns) * torch.exp(-best_hparams['tau_2_n'] * dists_tr_ns)
        scaling_factors_m = torch.exp(-best_hparams['tau_1_m'] * D_ms) * torch.exp(-best_hparams['tau_2_m'] * dists_tr_ms)

        d_ns = torch.sum(scaling_factors_n * dists_ns, dim=1) / D_ns.shape[1]
        d_ms = torch.sum(scaling_factors_m * dists_ms, dim=1) / D_ms.shape[1]

        scores = torch.tensor(df['d_1'].values) + best_hparams['beta'] * d_ns + best_hparams['gamma'] * d_ms
    else:
        D_ns = np.stack(df['D_n'].values)
        D_ms = np.stack(df['D_m'].values)
        dists_tr_ns = np.stack(df['dists_tr_n'].values)
        dists_tr_ms = np.stack(df['dists_tr_m'].values)
        dists_ns = np.stack(df['dists_n'].values)
        dists_ms = np.stack(df['dists_m'].values)

        scaling_factors_n = np.exp(-best_hparams['tau_1_n'] * D_ns) * np.exp(-best_hparams['tau_2_n'] * dists_tr_ns)
        scaling_factors_m = np.exp(-best_hparams['tau_1_m'] * D_ms) * np.exp(-best_hparams['tau_2_m'] * dists_tr_ms)

        d_ns = np.sum(scaling_factors_n * dists_ns, axis=1) / D_ns.shape[1]
        d_ms = np.sum(scaling_factors_m * dists_ms, axis=1) / D_ms.shape[1]

        scores = df['d_1'].values + best_hparams['beta'] * d_ns + best_hparams['gamma'] * d_ms

    if return_dn:
        return scores, d_ns, d_ms
    else:
        return scores      

def unpack_vector(x, force_zero = [], force_one = []):
    cand =  {
        'beta': x[0],
        'gamma': x[1],
        'tau_1_n': x[2],
        'tau_2_n': x[3],
        'tau_1_m': x[4],
        'tau_2_m': x[5]
    }

    for i in cand:
        if i in force_zero:
            cand[i] = 0.
    
    for i in cand:
        if i in force_one:
            cand[i] = 1.

    return cand

def approx_f1_loss(y_true, y_pred):
    tp = torch.sum((y_true * y_pred), dim=0)
    tn = torch.sum(((1 - y_true) * (1 - y_pred)), dim=0)
    fp = torch.sum(((1 - y_true) * y_pred), dim=0)
    fn = torch.sum((y_true * (1 - y_pred)), dim=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = 2 * p * r / (p + r + 1e-7)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1 - torch.mean(f1)

def optim_func(x, df, obj_func, obj_func_args, force_zero = [], force_one = []):
    hparams = unpack_vector(x, force_zero=force_zero, force_one=force_one)
    y = df['is_mislabel'].values
    score = calc_scores_given_hparams_vectorized(df, hparams, return_dn = False)
    return -obj_func(y, score, **obj_func_args)

def optim_func_torch(x, df, force_zero = [], force_one = []): # since F1 isn't differentiable, we need a proxy loss
    hparams = unpack_vector(x, force_zero=force_zero, force_one=force_one)
    y = df['is_mislabel'].values
    score = calc_scores_given_hparams_vectorized(df, hparams, return_dn = False, torch_arr = True)
    return nn.SoftMarginLoss()(score, torch.from_numpy(y).double() *2 - 1)

def torch_minimize(optim_func, x0, args, options={'max_iter': 20, 'line_search_fn': 'strong_wolfe'}):
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    optimizer = LBFGS([x], lr=0.1, max_iter=options['max_iter'], line_search_fn=options['line_search_fn'])
    def closure():
        optimizer.zero_grad()
        loss = optim_func(x, args[0])
        loss.backward()
        return loss
    
    for i in range(options['max_iter']):
        optimizer.step(closure)
    
    return {'x': x.detach().numpy(), 'fun': closure().item()}

def maximize_metric_scipy(df, x0, obj_func, obj_func_args, method, force_zero = [], force_one = []):
    return minimize(optim_func, x0, method = method,
                    args = (df, obj_func, obj_func_args, force_zero, force_one),
                   options={})

def maximize_metric_torch(df, x0,  obj_func, obj_func_args, force_zero = [], force_one = []):
    return torch_minimize(optim_func_torch, x0, args = (df, obj_func, obj_func_args, force_zero, force_one))
    
def maximize_metric(df, grid, x0s, obj_func, obj_func_args, force_zero = [], force_one = [], scipy_methods = ['Powell', 'Nelder-Mead']):
    best_x, best_val = None, -1
    for x in x0s:
        for method in scipy_methods:
            temp = maximize_metric_scipy(df, x, obj_func, obj_func_args, method = method, force_zero=force_zero, force_one=force_one)
            if -temp.fun > best_val:
                best_val = -temp.fun
                best_x = temp.x

    for x in x0s:
        cand_x = maximize_metric_torch(df, x, obj_func, obj_func_args, force_zero=force_zero, force_one=force_one)['x']
        temp = optim_func(cand_x, df, obj_func, obj_func_args, force_zero = force_zero, force_one=force_one)
        if -temp > best_val:
            best_val = -temp
            best_x = cand_x
    
    for x in combinations_base(grid):
        g = []
        for i in ['beta', 'gamma', 'tau_1_n', 'tau_2_n', 'tau_1_m', 'tau_2_m']:
            if i in x:
                g.append(x[i])
            else:
                if i in ['tau_1_n', 'tau_1_m']:
                    g.append(x['tau_1'])
                elif i in ['tau_2_n', 'tau_2_m']:
                    g.append(x['tau_2'])
                else:
                    raise NotImplementedError(i)

            if i in force_zero:
                g[-1] = 0.

        temp = optim_func(g, df, obj_func, obj_func_args, force_zero=force_zero, force_one=force_one)
        if -temp > best_val:
            best_val = -temp
            best_x = g

    for c, i in enumerate(['beta', 'gamma', 'tau_1_n', 'tau_2_n', 'tau_1_m', 'tau_2_m']):
        if i in force_zero:
            best_x[c] = 0.

        if i in force_one:
            best_x[c] = 1.
            
    score = calc_scores_given_hparams_vectorized(df, unpack_vector(best_x, force_zero=force_zero, force_one=force_one))
    return best_x, best_val, obj_func(df['is_mislabel'], score, return_thres = True, **obj_func_args)[1]

def cosDistance(features):
    # features: N*M matrix. N features, each features is M-dimension.
    features = F.normalize(features, dim=1) # each feature's l2-norm should be 1 
    similarity_matrix = torch.matmul(features, features.T)
    distance_matrix = 1.0 - similarity_matrix
    return distance_matrix

def count_knn_distribution(args, feat_cord, label, cluster_sum, k, norm='l2'):
    # feat_cord = torch.tensor(final_feat)
    KINDS = args.num_classes
    dist = cosDistance(feat_cord)

    print(f'knn parameter is k = {k}')
    time1 = time.time()
    min_similarity = args.min_similarity
    values, indices = dist.topk(k, dim=1, largest=False, sorted=True)
    values[:, 0] = 2.0 * values[:, 1] - values[:, 2]
    knn_labels = label[indices]

    knn_labels_cnt = torch.zeros(cluster_sum, KINDS)

    for i in range(KINDS):
        knn_labels_cnt[:, i] += torch.sum((1.0 - min_similarity - values) * (knn_labels == i), 1)

    time2 = time.time()
    print(f'Running time for k = {k} is {time2 - time1}')

    if norm == 'l2':
        # normalized by l2-norm -- cosine distance
        knn_labels_prob = F.normalize(knn_labels_cnt, p=2.0, dim=1)
    elif norm == 'l1':
        # normalized by mean
        knn_labels_prob = knn_labels_cnt / torch.sum(knn_labels_cnt, 1).reshape(-1, 1)
    else:
        raise NameError('Undefined norm')
    return knn_labels_prob

def get_metrics(pred_y, true_y, group_metrics):
    accuracies = []
    for c in group_metrics:
        accuracies += [(c, group_metrics[c]['accuracy'])]
    accuracies.sort(key=lambda x:x[1])
    return accuracy_score(true_y, pred_y), accuracies[0][1]

def get_group_metrics(pred, label, class_label, dataset_type):
    classes = np.unique(class_label)
    result = {}
    for c in classes:
        index = np.where(class_label == c)
        if dataset_type == "cifar10":
            class_name = cifar10_labels[c]
        elif dataset_type == "cifar100":
            class_name = cifar100_labels[c]
        else:
            class_name = str(c)
        result[class_name] = get_stats(label[index], pred[index])
    return result

def get_stats(true, pred):
    result = {}
    try:
        result['auroc'] = roc_auc_score(true, pred)
    except:
        result['auroc'] = None
    result['accuracy'] = accuracy_score(true, pred)
    if np.unique(true).size == 2:
        result['true_label_rate'] = np.mean(true)
        average = 'binary'
    else:
        average = 'micro'
    result['precision'] = precision_score(true, pred, labels=np.unique(true), average=average)
    result['f1_score'] = f1_score(true, pred, labels=np.unique(true), average=average)
    result['ece'] = ECE().measure(pred, true)
    return result

def optimize_f1(y, score, return_thres = False):
    best_thres, best_f1 = 0, 0
    for cand_thres in np.linspace(score.min(), score.max(), 100):
        pred_label = score >= cand_thres
        cand_f1 = f1_score(y, pred_label)
        if cand_f1 >= best_f1:
            best_f1 = cand_f1
            best_thres = cand_thres
    if return_thres:
        return best_f1, best_thres
    else:
        return best_f1
    
def optimize_f1_efficient(y, score, return_thres = False):
    def neg_f1(threshold):
        pred_label = score >= threshold
        return -f1_score(y, pred_label)
    best_thres = fminbound(neg_f1, score.min(), score.max(), xtol = 1e-8, disp = 0)
    best_f1 = -neg_f1(best_thres) 

    if return_thres:
        return best_f1, best_thres
    else:
        return best_f1

def f1_with_pred_prev_constraint(y, score, pred_prev, return_thres = False):
    f = lambda cand_thres: (score >= cand_thres).sum()/len(score) - pred_prev
    try:
        thres = bisect(f, score.min(), score.max())
        f1 = f1_score(y, score >= thres)
    except ValueError:
        return f1_with_pred_prev_constraint2(y, score, pred_prev, return_thres)
    
    if np.isnan(thres) or np.isnan(f1):
        return f1_with_pred_prev_constraint2(y, score, pred_prev, return_thres)

    if return_thres:
        return f1, thres
    else:
        return f1
    
def f1_with_pred_prev_constraint2(y, score, pred_prev, return_thres = False):
    f = lambda cand_thres: ((score >= cand_thres).sum()/len(score) - pred_prev)**2
    thres = fminbound(f, score.min(), score.max())
    f1 = f1_score(y, score >= thres)
    if return_thres:
        return f1, thres
    else:
        return f1

def derivative(f, a, h=0.01):
    return (f(a + h) - f(a - h))/(2*h)

def f1_with_local_minima_finder(y, score, return_thres = False):
    kde = gaussian_kde(score)
    x = np.linspace(score.min(), score.max(), 1000)
    y_kde = kde.evaluate(x)
    thress = x[argrelextrema(y_kde, np.less)]
    if len(thress) > 1:
        thres = np.median(thress)
    elif len(thress) == 1:
        thres = thress[0]
    else: # take median of local maxima  
        thress2 = x[argrelextrema(y_kde, np.greater)]
        if len(thress2) >= 2:
            thres = np.median(thress2)
        else: # fall back to global mean
            thres = np.mean(score)
    f1 = f1_score(y, score >= thres)

    if return_thres:
        return f1, thres
    else:
        return f1
    
def binary_metrics(targets, preds, label_set=[0, 1], suffix='', return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {
        'accuracy': accuracy_score(targets, preds),
        'F1': f1_score(targets, preds),
        'n_samples': len(targets)
    }

    if len(label_set) == 2:
        CM = confusion_matrix(targets, preds, labels=label_set)

        res['TN'] = CM[0][0].item()
        res['FN'] = CM[1][0].item()
        res['TP'] = CM[1][1].item()
        res['FP'] = CM[0][1].item()

        res['error'] = res['FN'] + res['FP']

        if res['TP'] + res['FN'] == 0:
            res['TPR'] = 0
            res['FNR'] = 1
        else:
            res['TPR'] = res['TP']/(res['TP']+res['FN'])
            res['FNR'] = res['FN']/(res['TP']+res['FN'])

        if res['FP'] + res['TN'] == 0:
            res['FPR'] = 1
            res['TNR'] = 0
        else:
            res['FPR'] = res['FP']/(res['FP']+res['TN'])
            res['TNR'] = res['TN']/(res['FP']+res['TN'])

        if res['TP'] + res['FP'] > 0:
            res['PPV'] = res['TP'] / (res['TP'] + res['FP'])
        else:
            res['PPV'] = 0  

        if res['TN'] + res['FN'] > 0:
            res['NPV'] = res['TN'] / (res['TN'] + res['FN'])
        else:
            res['NPV'] = 0  

        res['pred_prevalence'] = (res['TP'] + res['FP']) / res['n_samples']
        res['prevalence'] = (res['TP'] + res['FN']) / res['n_samples']
    else:
        CM = confusion_matrix(targets, preds, labels=label_set)
        res['TPR'] = recall_score(targets, preds, labels=label_set, average='macro', zero_division=0.)

    if len(np.unique(targets)) > 1:
        res['balanced_acc'] = balanced_accuracy_score(targets, preds)

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return {f"{i}{suffix}": res[i] for i in res}


def prob_metrics(targets, preds, sample_weight = None):
    return {
        'AUROC': roc_auc_score(targets, preds, sample_weight=sample_weight),
        'AUPRC': average_precision_score(targets, preds, average='macro', sample_weight=sample_weight)
    }

def eval_metrics(y, score, prevalence, fix_thress = {}, use_efficient = False):
    if 'F1_optimal_thres' in fix_thress:
        f1_optim_thres = fix_thress['F1_optimal_thres']
    else:
        if use_efficient:
            f1_optim, f1_optim_thres = optimize_f1_efficient(y, score, True)
        else:
            f1_optim, f1_optim_thres = optimize_f1(y, score, True)

    if 'F1_prev_thres' in fix_thress:
        f1_prev_thres = fix_thress['F1_prev_thres']
    else:
        f1_prev, f1_prev_thres = f1_with_pred_prev_constraint(y, score, prevalence, True)

    if 'F1_heuristic_thres' in fix_thress:
        f1_heuristic_thres = fix_thress['F1_heuristic_thres']
    else:
        f1_heuristic, f1_heuristic_thres = f1_with_local_minima_finder(y, score, True)

    return {**prob_metrics(y, score), **{
        'F1_optimal_thres': f1_optim_thres,
        'F1_prev_thres': f1_prev_thres,
        'F1_heuristic_thres': f1_heuristic_thres
        },
        **binary_metrics(y, score >= f1_optim_thres, suffix = '_optimal'),
        **binary_metrics(y, score >= f1_prev_thres, suffix = '_prev'),
        **binary_metrics(y, score >= f1_heuristic_thres, suffix = '_heuristic')
    }