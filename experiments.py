import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import json
from tqdm import tqdm
import pickle
import copy

def combinations_base(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))

def combinations(grid):
    sub_exp_names = set()
    args = []
    for i in grid:
        if isinstance(grid[i], dict):
            for j in grid[i]:
                sub_exp_names.add(j)
    if len(sub_exp_names) == 0:
        return combinations_base(grid)

    for i in grid:
        if isinstance(grid[i], dict):
            assert (
                set(list(grid[i].keys())) == sub_exp_names
            ), f"{i} does not have all sub exps ({sub_exp_names})"
    for n in sub_exp_names:
        sub_grid = grid.copy()
        for i in sub_grid:
            if isinstance(sub_grid[i], dict):
                sub_grid[i] = sub_grid[i][n]
        args += combinations_base(sub_grid)
    return args


def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment]().get_hparams()


def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].fname


    
class multimodal_knn_caption:
    fname = "run_lemon"

    def __init__(self):
        self.hparams = {
            "dataset": {
                'exp1': ['mscoco', 'mmimdb'],
                'exp2': ['flickr30k'],
                'exp3': ['mimiccxr_caption'],
                'exp4': ['cifar10', 'cifar100']
            },
            "dist_type": ['euclidean', 'cosine'],
            'normalize_d1': [False],
            'noise_type': {
                'exp1': ['random', 'cat', 'noun'],
                'exp2': ['random', 'noun'],
                'exp3': ['random', 'cat'],
                'exp4': ['real', 'symmetric', 'asymmetric']
            },
            'clip_model': {
                'exp1': ['huggingface_clip'],
                'exp2': ['huggingface_clip'],
                'exp3': ['biomed_clip'],
                'exp4': ['huggingface_clip']
            },
            "noise_level": [0.4],
            'ablation': ['none', 'multimodal_baseline'],
            'custom_cifar_prompt': {
                'exp1': [''],
                'exp2': [''],
                'exp3': [''],
                'exp4': ['', 'A photo of a '],
            },
            'knn_k': [1, 2, 5, 10, 15, 20, 30, 50],
            "data_seed": [0, 1, 2],
            'use_discrete_for_text':{
                'exp1': [False],
                'exp2': [False],
                'exp3': [False],
                'exp4': [True],
            }
        }

    def get_hparams(self):
        return combinations(self.hparams)

class multimodal_knn_caption_real:
    fname = "run_lemon"

    def __init__(self):
        self.hparams = {
            "dataset": {
                'exp1': ['mscoco'],
                'exp2': ['flickr30k'],
                'exp3': ['mimiccxr_caption'],
                'exp4': ['cifar10', 'cifar100']
            },
            "dist_type": ['cosine'], 
            'normalize_d1': [False],
            'real_dataset': [True],
            'noise_type': {
                'exp1': ['random'],
                'exp2': ['random'],
                'exp3': ['random'],
                'exp4': ['asymmetric']
            },
            'clip_model': {
                'exp1': ['huggingface_clip'],
                'exp2': ['huggingface_clip'],
                'exp3': ['biomed_clip'],
                'exp4': ['huggingface_clip']
            },
            'use_discrete_for_text': {
                'exp1': [False],
                'exp2': [False],
                'exp3': [False],
                'exp4': [True]
            },
            "noise_level": [0.0],
            'ablation': ['none', 'multimodal_baseline'],
            'knn_k': [30], 
            "data_seed": [0],
        }

    def get_hparams(self):
        return combinations(self.hparams)



class multimodal_knn_caption_vary_noise:
    fname = "run_lemon"

    def __init__(self):
        self.hparams1 = {
            "dataset": {
                'exp1': ['mscoco', 'mmimdb'],
                'exp2': ['flickr30k'],
            },
            "dist_type": ['euclidean', 'cosine'],
            'normalize_d1': [False],
            'noise_type': {
                'exp1': ['cat'],
                'exp2': ['noun'],
            },
            'clip_model': {
                'exp1': ['huggingface_clip'],
                'exp2': ['huggingface_clip'],
            },
            "noise_level": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'ablation': ['none'],
            'knn_k': [1, 2, 5, 10, 15, 20, 30, 50],
            "skip_train": [True],
            "data_seed": [0, 1, 2],
            'skip_hparam_optim': [True]
        }

        self.hparams2 = {
            "dataset": {
                'exp1': ['mscoco', 'mmimdb'],
                'exp2': ['flickr30k'],
            },
            "dist_type": ['euclidean', 'cosine'],
            'normalize_d1': [False],
            'noise_type': {
                'exp1': ['cat'],
                'exp2': ['noun'],
            },
            'clip_model': {
                'exp1': ['huggingface_clip'],
                'exp2': ['huggingface_clip'],
            },
            "noise_level": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'ablation': ['multimodal_baseline'],
            'knn_k': [1],
            "skip_train": [True],
            "data_seed": [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams1) + combinations(self.hparams2)
    

class multimodal_knn_caption_ablations:
    fname = "run_lemon"

    def __init__(self):
        self.hparams = {
            "dataset": {
                'exp1': ['mscoco', 'mmimdb'],
            },
            "dist_type": ['euclidean', 'cosine'],
            'normalize_d1': [False],
            'noise_type': {
                'exp1': ['cat'],
            },
            'clip_model': {
                'exp1': ['huggingface_clip'],
            },
            "noise_level": [0.4],
            'ablation':  ['none', 'tau_1', 'tau_2', 'tau_1_2', 'beta', 'gamma',
                                                                'multimodal_baseline'],
            'knn_k': [1, 2, 5, 10, 15, 20, 30, 50],
            "skip_train": [True],
            "data_seed": [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams)

class multimodal_knn_cifar_ablations:
    fname = "run_lemon"

    def __init__(self):
        self.hparams = {
            "dataset": ['cifar100'],
            "dist_type": ['euclidean', 'cosine'],
            'normalize_d1': [False],
            'noise_type': ['real'],
            'clip_model': ['huggingface_clip'],
            "noise_level": [0.4],
            'ablation':  ['none', 'tau_1', 'tau_2', 'tau_1_2', 'beta', 'gamma',
                                                                'multimodal_baseline'],
            'knn_k': [1, 2, 5, 10, 15, 20, 30, 50],
            "skip_train": [True],
            "data_seed": [0, 1, 2],
            'use_discrete_for_text': [True],
            'custom_cifar_prompt': ['', 'A photo of a '],
        }

    def get_hparams(self):
        return combinations(self.hparams)
    
    
class multimodal_knn_caption_mimic_clip_scratch:
    fname = "run_lemon"

    def __init__(self):
        self.hparams = {
            "dataset": ['mimiccxr_caption'],
            "dist_type": ['euclidean', 'cosine'],
            'normalize_d1': [False],
            'noise_type': {
                'exp1': ['random'],
                'exp2': ['cat']
            },
            'clip_model': {
                'exp1': ['mimic_clip_from_scratch_random', 'chexzero'],
                'exp2': ['mimic_clip_from_scratch_cat', 'chexzero']
            },
            "noise_level": [0.4],
            'ablation': ['none', 'multimodal_baseline'],
            'knn_k': [1, 2, 5, 10, 15, 20, 30, 50],
            "data_seed": [0, 1, 2],
            'skip_hparam_optim': [True]
        }

    def get_hparams(self):
        return combinations(self.hparams)
    