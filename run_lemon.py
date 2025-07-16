import argparse
import collections
import json
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import PIL
import pickle
from transformers import AutoTokenizer
from scipy.special import softmax
import faiss
import socket
from pathlib import Path
from tqdm import tqdm

import torch
import torchvision
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.models.utils import get_img_base, algorithm_class_from_scratch
from lib.datasets.utils import get_dataset, cifar10_labels, cifar100_labels, mini_imagenet_labels, stanford_cars_labels
from lib.utils.utils import path_serial, Tee, normalize_vectors
from lib.metrics import utils
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
clf_datasets = ["cifar10", "cifar100", 'cifar10_full','cifar100_full', 'mini_imagenet', 'stanford_cars']

parser = argparse.ArgumentParser(description="LEMoN")
parser.add_argument("--exp_name", type=str)
parser.add_argument('--output_dir', type = str, required = True)
parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", 'flickr30k', 'mscoco', 'mimiccxr_caption', 'mmimdb', 'cifar10_full','cifar100_full', 
                                                                        'mini_imagenet', 'stanford_cars', 'cc3m'])
parser.add_argument("--noise_type", type=str, default="real", choices=["real", "asymmetric", "symmetric", "random", "noun", "cat"])
parser.add_argument("--noise_level", type=float, default = 0.4)
parser.add_argument("--dist_type", type=str, default="cosine", choices=["cosine", "euclidean"])
parser.add_argument('--normalize_d1', action = 'store_true', help = 'normalize CLIP sim by all possible labels. Only for CIFAR-10 and CIFAR-100')
parser.add_argument("--clip_model", type=str, default="huggingface_clip", choices = ["huggingface_clip", 'biomed_clip', 'mimic_clip_from_scratch_random', 'mimic_clip_from_scratch_cat', 'chexzero', 'cc3m_clip_from_scratch'])
parser.add_argument('--knn_k', default = 5, type = int)
parser.add_argument('--batch_size', default = 128, type = int)
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--data_seed', default = 0, type = int)
parser.add_argument('--compr_dataset_size_limit', default = 50000, type = int)
parser.add_argument('--ablation', default = 'none', choices = ['none', 'tau_1', 'tau_2', 'tau_1_2', 'beta', 'gamma',
                                                                'multimodal_baseline', 'd1', 'only_gamma', 'only_beta'])
parser.add_argument('--use_discrete_for_text', action = 'store_true', help = 'use the discrete metric for text comparisons')
parser.add_argument('--real_dataset', action = 'store_true', help = 'Running on real dataset, do not optimize hparams')
parser.add_argument('--custom_cifar_prompt', default = None)
parser.add_argument('--subset_val_set', default = -1, type = int)
parser.add_argument('--debug', action = 'store_true')
parser.add_argument('--skip_train', action = 'store_true')
parser.add_argument('--skip_hparam_optim', action = 'store_true')
args = parser.parse_args()
hparams = vars(args)

out_dir = Path(args.output_dir)
out_dir.mkdir(exist_ok = True, parents = True)

if not args.debug:
    sys.stdout = Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output_dir, 'err.txt'))

print("Environment:")
print("\tPython: {}".format(sys.version.split(" ")[0]))
print("\tPyTorch: {}".format(torch.__version__))
print("\tCUDA: {}".format(torch.version.cuda))
print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
print("\tNumPy: {}".format(np.__version__))
print("\tNode: {}".format(socket.gethostname()))

print('Args:')
for k, v in sorted(hparams.items()):
    print('\t{}: {}'.format(k, v))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(out_dir/'args.json', 'w') as outfile:
    json.dump(vars(args), outfile, default=path_serial)

if hparams['real_dataset']:
    assert hparams['noise_level'] == 0.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if hparams['dataset'] in clf_datasets:
    label_set = {
        'cifar10': cifar10_labels,
        'cifar10_full': cifar10_labels,
        'cifar100': cifar100_labels,
        'cifar100_full': cifar100_labels,
        'stanford_cars': stanford_cars_labels,
        'mini_imagenet': mini_imagenet_labels
    }[hparams['dataset']]
else:
    label_set = None

train_set, val_set, test_set = get_dataset(hparams['dataset'], args.data_seed, noisy_labels = True, percent_flips=args.noise_level, 
                                        flip_type=args.noise_type, return_combined_dataset = True)

if args.subset_val_set > 0:
    rng = np.random.default_rng(args.data_seed)
    val_set = torch.utils.data.Subset(val_set, indices = rng.choice(np.arange(len(val_set)), min(args.subset_val_set, len(val_set)), replace = False))

algorithm, tokenizer = algorithm_class_from_scratch(
    args.clip_model, text_base_name='openai/clip-vit-base-patch32', img_base=None, return_tokenizer=True
)
algorithm = algorithm.eval().to(device)

def prompt_fn_generator_simple(prefix):
    return lambda x: prefix + x
prompt_fn = prompt_fn_generator_simple('A photo of a ' if args.custom_cifar_prompt is None else args.custom_cifar_prompt)

# embed train set
if len(train_set) > args.compr_dataset_size_limit:
    train_indices_in_compr = np.random.choice(np.arange(len(train_set)), args.compr_dataset_size_limit, replace = False)
    compr_set = torch.utils.data.Subset(train_set, train_indices_in_compr)
else:
    train_indices_in_compr = np.arange(len(train_set))
    compr_set = train_set

dataloader = DataLoader(
        dataset=compr_set, batch_size=hparams['batch_size'], num_workers=8
)
bs = hparams['batch_size']
k = hparams['knn_k']

start_t = datetime.now()
emb_img, emb_txt, tr_text_labels = [], [], []
for idx, batch in enumerate(dataloader):
    pixel_values = batch[0].to(device)

    if hparams['dataset'] in clf_datasets:
        noisy_labels = batch[2]
        noisy_text_labels = label_set[noisy_labels].tolist()
        text_labels = [prompt_fn(i) for i in noisy_text_labels]
    else:
        text_labels = batch[2]    
    tr_text_labels += text_labels
    
    if args.clip_model == 'biomed_clip' or args.clip_model.startswith('mimic_clip_from_scratch') or args.clip_model == 'chexzero' or args.clip_model == 'cc3m_clip_from_scratch':
        encodings = tokenizer(text_labels).to(device)
    else:
        encodings = tokenizer(
                text_labels, padding="max_length", truncation=True)
        input_ids = torch.tensor(encodings["input_ids"]).to(device)
        attention_mask = torch.tensor(encodings["attention_mask"]).to(device)

    with torch.no_grad():
        if args.clip_model == 'biomed_clip' or args.clip_model.startswith('mimic_clip_from_scratch') or args.clip_model == 'chexzero' or args.clip_model == 'cc3m_clip_from_scratch':
            emb_txt.append(algorithm.encode_text(encodings).detach().cpu())
        else:
            emb_txt.append(algorithm.encode_text(input_ids, attention_mask).detach().cpu())
        emb_img.append(algorithm.encode_image(pixel_values).detach().cpu())
    
emb_txt_tr = normalize_vectors(torch.concat(emb_txt))
emb_img_tr = normalize_vectors(torch.concat(emb_img))

if hparams['dist_type'] == 'cosine':
    index_txt = faiss.IndexFlatIP(emb_txt_tr.shape[1])
    index_img = faiss.IndexFlatIP(emb_img_tr.shape[1])
    dists_tr = 1 - (emb_txt_tr * emb_img_tr).sum(axis = 1)
elif hparams['dist_type'] == 'euclidean':
    index_txt = faiss.IndexFlatL2(emb_txt_tr.shape[1])
    index_img = faiss.IndexFlatL2(emb_img_tr.shape[1])
    dists_tr = ((emb_txt_tr - emb_img_tr)**2).sum(axis = 1)

index_txt.add(emb_txt_tr.numpy())
index_img.add(emb_img_tr.numpy())
tr_text_labels = np.array(tr_text_labels)

# all dataset labels, used for normalization
if hparams['dataset'] in clf_datasets:
    dataset_prompt_labels = [prompt_fn(i) for i in label_set]
    if args.clip_model == 'biomed_clip' or args.clip_model.startswith('mimic_clip_from_scratch') or args.clip_model == 'chexzero' or args.clip_model == 'cc3m_clip_from_scratch':
        encodings = tokenizer(dataset_prompt_labels).to(device)
        text_embeds_dataset_labels = normalize_vectors(algorithm.encode_text(encodings).detach().cpu())
    else:
        encodings = tokenizer(
                dataset_prompt_labels, padding="max_length", truncation=True)
        input_ids = torch.tensor(encodings["input_ids"]).to(device)
        attention_mask = torch.tensor(encodings["attention_mask"]).to(device)
        text_embeds_dataset_labels = normalize_vectors(algorithm.encode_text(input_ids, attention_mask).detach().cpu())

logs = []
if args.debug or args.skip_train:
    sets_iter = zip(['val', 'test'],  [val_set, test_set])
else:
    sets_iter = zip(['train', 'val', 'test'], [train_set, val_set, test_set])

for sname, dset in sets_iter:
    dataloader = DataLoader(
        dataset=dset, batch_size=bs, num_workers=8
    )
    for idx, batch in tqdm(enumerate(dataloader), total = len(dataloader)):
        noisy_labels = batch[2]
        real_labels = batch[1]
        pixel_values = batch[0].to(device)

        if hparams['dataset'] in clf_datasets:
            noisy_text_labels = label_set[noisy_labels].tolist()        
            clean_text_labels = label_set[real_labels].tolist()
            noisy_text_labels_prompts = [prompt_fn(i) for i in noisy_text_labels]
        else:
            noisy_text_labels = noisy_labels
            noisy_text_labels_prompts = noisy_labels
            clean_text_labels = real_labels
                
        label_flips = np.array(noisy_text_labels)==np.array(clean_text_labels)
        label_flips = 1-label_flips
        
        if args.clip_model == 'biomed_clip' or args.clip_model.startswith('mimic_clip_from_scratch') or args.clip_model == 'chexzero' or args.clip_model == 'cc3m_clip_from_scratch':
            encodings = tokenizer(noisy_text_labels_prompts).to(device)
        else:
            encodings = tokenizer(
                    noisy_text_labels_prompts, padding="max_length", truncation=True)
            
            input_ids = torch.tensor(encodings["input_ids"]).to(device)
            attention_mask = torch.tensor(encodings["attention_mask"]).to(device)

        with torch.no_grad():
            if args.clip_model == 'biomed_clip' or args.clip_model.startswith('mimic_clip_from_scratch') or args.clip_model == 'chexzero' or args.clip_model == 'cc3m_clip_from_scratch':
                text_embeds = normalize_vectors(algorithm.encode_text(encodings).detach().cpu())
            else:
                text_embeds = normalize_vectors(algorithm.encode_text(input_ids, attention_mask).detach().cpu())
            img_embeds = normalize_vectors(algorithm.encode_image(pixel_values).detach().cpu())

        D_ns, I_ns = index_img.search(img_embeds.numpy(), k + (sname == 'train'))
        D_ms, I_ms = index_txt.search(text_embeds.numpy(), k+ (sname == 'train'))

        for i in range(len(img_embeds)):
            sample_idx = idx * bs + i
            img_embed = img_embeds[i, None]
            text_embed = text_embeds[i, None]

            # d_1
            if args.normalize_d1:
                if hparams['dist_type'] == 'cosine':
                    d1 = softmax(1 - (img_embed * text_embeds_dataset_labels).sum(axis = 1))[noisy_labels[i]]
                elif hparams['dist_type'] == 'euclidean':
                    d1 = softmax(((img_embed.flatten() - text_embeds_dataset_labels)**2).sum(axis = 1))[noisy_labels[i]]
            else:
                if hparams['dist_type'] == 'cosine':
                    d1 = 1 - torch.dot(img_embed.flatten(), text_embed.flatten())
                elif hparams['dist_type'] == 'euclidean':
                    d1 = ((img_embed.flatten() - text_embed.flatten())**2).sum()

            # d_n
            D_n, I_n = D_ns[i], I_ns[i]
            if sname == 'train': # skip over same sample
                if sample_idx in train_indices_in_compr:
                    I_n = I_n[1:]
                    D_n = D_n[1:]
                else:
                    I_n = I_n[:-1]
                    D_n = D_n[:-1]
            y_n = emb_txt_tr[I_n]

            if args.use_discrete_for_text:
                dists_n = 1 - torch.Tensor(tr_text_labels[I_n] == noisy_text_labels_prompts[i]).float()
            else:
                if hparams['dist_type'] == 'cosine':
                    D_n = -D_n
                    dists_n = 1 - (text_embed * y_n).sum(axis = 1)
                elif hparams['dist_type'] == 'euclidean':
                    dists_n = ((text_embed - y_n)**2).sum(axis = 1)

            # d_m
            D_m, I_m = D_ms[i], I_ms[i]
            if sname == 'train': # skip over same sample
                if sample_idx in train_indices_in_compr:
                    I_m = I_m[1:]
                    D_m = D_m[1:]
                else:
                    I_m = I_m[:-1]
                    D_m = D_m[:-1]
            x_m = emb_img_tr[I_m]
            if hparams['dist_type'] == 'cosine':
                D_m = -D_m
                dists_m = 1 - (img_embed * x_m).sum(axis = 1)
            elif hparams['dist_type'] == 'euclidean':
                dists_m = ((img_embed - x_m)**2).sum(axis = 1)

            logs.append({
                'sset': sname,
                'idx': sample_idx,
                'actual_label': real_labels[i].item() if torch.is_tensor(real_labels[i]) else real_labels[i],
                'actual_label_text': clean_text_labels[i],
                'noisy_label': noisy_labels[i],
                'noisy_label_text': noisy_text_labels[i],
                'is_mislabel': label_flips[i],
                'is_correct_label': 1 - label_flips[i],
                'd_1': d1.item(),
                'dists_n': dists_n.numpy(),
                'D_n': D_n.flatten(),
                'dists_tr_n': dists_tr[I_n].numpy(),
                'dists_m': dists_m.numpy(),
                'D_m': D_m.flatten(),
                'dists_tr_m': dists_tr[I_m].numpy()
            })

end_t = datetime.now()
timedelta = (end_t - start_t).total_seconds() 
n_samples = len(logs)
print(f"Finished {n_samples} samples in {timedelta} seconds; avg of {timedelta/n_samples}s per sample")

df = pd.DataFrame(logs)

if 'd1' in args.ablation:
    df['d_1'] = 0.0

if args.real_dataset or args.skip_hparam_optim:
    res = {
        'df': df
    }
else:    
    df_val = df.query('sset == "val"')
    obj_funcs = {
        'know_val_labels': utils.optimize_f1_efficient,
    }
    side_info = {
        'know_val_labels': {},
    }

    grid = { 
        'beta': np.arange(0, 100.01, 5),
        'gamma': np.arange(0, 100.01, 5),
        'tau_1': [0, 1, 5, 10], # = tau_1_n = tau_2_n
        'tau_2': [0, 1, 5, 10],
    }

    selection_results = {}
    for selection_criteria in obj_funcs: # compute optimal beta and gamma to get a score
        if args.ablation == 'only_beta':
            selection_results[selection_criteria] = {
                'beta': 1,
                'gamma': 0,
                'tau_1_n': 0, 
                'tau_2_n': 0, 
                'tau_1_m': 0, 
                'tau_2_m': 0,
            }
        elif args.ablation == 'only_gamma':
            selection_results[selection_criteria] = {
                'beta': 0,
                'gamma': 1,
                'tau_1_n': 0, 
                'tau_2_n': 0, 
                'tau_1_m': 0, 
                'tau_2_m': 0,
            }
        else:
            if args.ablation == 'multimodal_baseline':
                best_beta, best_gamma, best_tau_1_n, best_tau_2_n, best_tau_1_m, best_tau_2_m = [0] * 6
                best_f1, best_thres = obj_funcs[selection_criteria](df_val['is_mislabel'], df_val['d_1'], return_thres = True, **side_info[selection_criteria])
            else:
                if args.ablation == 'none' or args.ablation == 'd1':
                    force_zero = []
                elif args.ablation == 'tau_1':
                    force_zero = ['tau_1_n', 'tau_1_m']
                elif args.ablation == 'tau_2':
                    force_zero = ['tau_2_n', 'tau_2_m']
                elif args.ablation == 'tau_1_2':
                    force_zero = ['tau_1_n', 'tau_1_m', 'tau_2_n', 'tau_2_m']
                elif args.ablation in ['beta', 'd1_beta']:
                    force_zero = ['beta']
                elif args.ablation in ['gamma', 'd1_gamma']:
                    force_zero = ['gamma']

                if args.ablation == 'd1':
                    force_one = ['beta'] # arbitrary to remove a degree of freedom
                elif args.ablation == 'd1_beta':
                    force_one = ['gamma']
                elif args.ablation == 'd1_gamma':
                    force_one = ['beta']
                else:
                    force_one = []

                (best_beta, best_gamma, best_tau_1_n, best_tau_2_n, best_tau_1_m, best_tau_2_m), best_f1, best_thres = utils.maximize_metric(
                    df_val,
                    grid,
                    [[0] * 6, [0.5] * 6, [1] * 6, [10] * 6],
                    obj_funcs[selection_criteria],
                    side_info[selection_criteria],
                    force_zero = force_zero,
                    force_one = force_one
                )
            selection_results[selection_criteria] = {
                'beta': best_beta,
                'gamma': best_gamma,
                'thres': best_thres,
                'tau_1_n': best_tau_1_n, 
                'tau_2_n': best_tau_2_n, 
                'tau_1_m': best_tau_1_m, 
                'tau_2_m': best_tau_2_m,
                'selected_val': best_f1
            }

        (df[f'{selection_criteria}_pred_score'], df[f'{selection_criteria}_d_n'], 
            df[f'{selection_criteria}_d_m']) = utils.calc_scores_given_hparams_vectorized(df, selection_results[selection_criteria], True)

        df_val = df.query('sset == "val"')
        thress = utils.eval_metrics(df_val['is_mislabel'],
                                    df_val[f'{selection_criteria}_pred_score'],
                                    prevalence = df.loc[df.sset == 'val', 'is_mislabel'].sum()/(df.sset == 'val').sum())

        for sset in df.sset.unique(): # eval score on each set
            sub_df = df.loc[df.sset == sset]
            selection_results[selection_criteria][sset] = utils.eval_metrics(sub_df['is_mislabel'],
                                                                        sub_df[f'{selection_criteria}_pred_score'],
                                                                        prevalence = df.loc[df.sset == 'val', 'is_mislabel'].sum()/(df.sset == 'val').sum(),
                                                                        fix_thress = thress)
        df[['sset', 'idx', 'actual_label', 'noisy_label', 'is_mislabel', f'{selection_criteria}_pred_score']].rename(columns = {
            f'{selection_criteria}_pred_score': 'pred_score'
        }).to_csv(out_dir/f'{selection_criteria}_scores.csv')
        
    res = {
        'df': df,
        'agg_results': selection_results
    }
    
pickle.dump(res, (out_dir/'res.pkl').open('wb'))

if args.skip_hparam_optim:
    with open(os.path.join(out_dir, 'need_hparam_optim'), 'w') as f:
        f.write('need_hparam_optim')

with open(os.path.join(out_dir, 'done'), 'w') as f:
    f.write('done')
