import numpy as np
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

import torch

from lib.metrics.utils import count_knn_distribution

metric_names = [
    "tpr",
    "tnr",
    "fpr",
    "fnr",
    "fdr",
    "ppv",
    "f1",
    "auc",
    "apr",
    "acc",
    "loss",
]
score_to_dict = lambda name, score: dict((name[i], score[i]) for i in range(len(score)))


class DistanceEvaluator(object):
    '''Evaluator Object for 
    distance performance'''

    def __init__(self, y_true, y_pred_proba, dist='cosine',threshold=0.5, y_pred_prob_epochs=None, loss=None, 
                 first_modality_embeddings=None, second_modality_embeddings=None):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.loss = loss
        self.threshold = threshold
        self.first_modality_embeddings = first_modality_embeddings
        self.second_modality_embeddings = second_modality_embeddings
        self.y_pred_prob_epochs = y_pred_prob_epochs
        self.dist = dist

    def our_metric(self):
        """Compute AUC and accuracy

        Returns:
            auc: auc for objective clip classification
            accuracy: accuracy for objective clip classification
        """
        #NB: These metrics first compute all pairwise metrics, then
        # we just take the diagonal or the self-pairs
        if self.dist == "cosine":
            return 1 - np.diagonal(cosine_similarity(
                self.first_modality_embeddings.cpu().data.numpy(),
                self.second_modality_embeddings.cpu().data.numpy(),
            ))
        elif self.dist == "euclidean":
            return np.diagonal(euclidean_distances(
                self.first_modality_embeddings.cpu().data.numpy(),
                self.second_modality_embeddings.cpu().data.numpy(),
            ))
        elif self.dist == "manhattan":
            return np.diagonal(manhattan_distances(
                self.first_modality_embeddings.cpu().data.numpy(),
                self.second_modality_embeddings.cpu().data.numpy(),
            ))
        else:
            raise NotImplementedError

    def clf_performance(self, y_true_curr=None, y_pred_proba_curr=None):
        """Compute AUC and accuracy

        Returns:
            auc: auc for objective clip classification
            accuracy: accuracy for objective clip classification
        """

        if y_true_curr is not None and y_pred_proba_curr is not None:
            roc = roc_auc_score(y_true_curr, y_pred_proba_curr)
            acc = accuracy_score(y_true_curr, y_pred_proba_curr > self.threshold)
        
        else: 
            roc = roc_auc_score(self.y_true, self.y_pred_proba)
            acc = accuracy_score(self.y_true, self.y_pred_proba > self.threshold)

        return roc, acc

    def get_datamap_score(self, datamap_threshold=0.2,thresholding=False):

        """
        Unimodal baseline
        Parameters:
            y_pred_prob_epochs: Vector of size N X C X Epochs,
                where N=size of train set, C=Number of classes,
                Epochs=number of training epochs
            y_true: True class, 0-indexed


        Returns:
            datamap scores, np.array of floats.
        """
        try:
            assert np.min(self.y_true) == 0
        except:
            raise NotImplementedError

        # getting scores over epochs
        instance_arr = []

        for i in range(len(self.y_true)):
            true_class_probs = self.y_pred_prob_epochs[:, i, self.y_true[i]]
            instance_arr.append(true_class_probs)

        mean_scores = np.mean(instance_arr, axis=1)
        var_scores = np.std(instance_arr, axis=1)

        datamap_scores = []
        datamap_tuple_scores = []
        for i in range(len(self.y_true)):
            curr_score = 0
            mean_score = mean_scores[i]
            var_score = var_scores[i]
            curr_score = (mean_score<0.5) and (var_score<0.1)
            datamap_tuple_scores.append([mean_score, var_score])
            if thresholding:
                if mean_score <= datamap_threshold and var_score <= datamap_threshold:
                    curr_score = 1
                else:
                    curr_score = 0

            datamap_scores.append(curr_score)
        return datamap_scores, np.array(datamap_tuple_scores)

    def get_aum_score(self, y_true, aum_threshold=0.2, thresholding=False):
        """
        Parameters:
            y_pred_prob_epochs: Vector of size N X C X Epochs,
                where N=size of train set, C=Number of classes,
                Epochs=number of training epochs
            y_true: True class, 0-indexed


        Returns:
            AUM scores, np.array of floats.
        """
        try:
            assert np.min(y_true) == 0
        except:
            raise NotImplementedError

        # getting scores over epochs
        aum_scores = []
        for i in range(len(self.y_true)):
            true_class_probs = self.y_pred_prob_epochs[:, i, y_true[i]]
            curr_margins = []
            for epoch in range(self.y_pred_prob_epochs.shape[0]):
                all_other_inds = list(set(y_true.tolist())-set([y_true[i]]))
                curr_margins.append(
                    true_class_probs[epoch] - np.max(self.y_pred_prob_epochs[epoch, i, all_other_inds]))
                

            if thresholding:
                aum_scores.append(np.mean(curr_margins) > aum_threshold)
            else:
                aum_scores.append(np.mean(curr_margins))

        return aum_scores


    def confidence_interval(self, values, alpha=0.95):
        lower = np.percentile(values, (1 - alpha) / 2 * 100)
        upper = np.percentile(values, (alpha + (1 - alpha) / 2) * 100)
        return lower, upper

    def return_pred(self):
        return self.y_pred_proba>self.threshold, self.y_true
    
