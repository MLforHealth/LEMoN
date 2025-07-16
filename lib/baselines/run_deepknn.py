"""
Script to test MultiModal-NeighBour (NNMB) methods 
"""

import argparse
import collections
import json
import os
import random
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import PIL
import pickle
import torch
import torchvision
import torch.utils.data
import torch.optim as optim
from tensorboard_logger import Logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lib.utils.utils import EarlyStopping
from lib.models.utils import get_img_base, algorithm_class_from_scratch
from lib.datasets.utils import get_dataset
from lib.metrics.distance_metrics import DistanceEvaluator
from lib.metrics.multimodal_neighbor_v2 import MMNB
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


import copy

from torch import nn
import faiss
import time
  


MAX_SIZE=1000
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal distance metric")
    parser.add_argument("--exp_name", type=str, default="pretrain")

    # training
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["imagenet", "cifar10", "cifar100",
                 "mscoco","flickr30k",
                 "mmimdb","mimiccxr_caption",
                 "stanford_cars","mini_imagenet",
                 ],
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="huggingface_clip",
        choices=["huggingface_clip", "medclip", "finetune","biomed_clip"],
    )
    parser.add_argument("--text_base_name", type=str, 
                        # default="albert-base-v2"
                        default='openai/clip-vit-base-patch32'
                        )
    
    parser.add_argument(
        "--img_base_name",
        type=str,
        default="clipvisionmodel",
        choices=["clipvisionmodel", "clipvisionmodelvit"],
    )
    
    parser.add_argument("--output_folder_name", type=str, default="./")

    # others
    parser.add_argument("--data_dir", type=str, default="/data/scratch/aparnab/")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument(
        "--hparams_seed",
        type=int,
        default=0,
        help='Seed for random hparams (0 for "default hparams")',
    )
    
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")

    # noise type
    parser.add_argument("--flip_type", type=str, default="real")

    # hparams
    parser.add_argument("--batch_size", default=258, type=int)
    parser.add_argument("--epochs", default=20)
    parser.add_argument("--percent_flips", type=float, default=0.3)
    parser.add_argument("--noise_labels", action='store_true')
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--dist_type", type=str, default="cosine")
    parser.add_argument("--method", type=str, default="ours")

    # checkpoints
    parser.add_argument("--val_only", action="store_true")
    parser.add_argument("--store_name", type=str, default=".")
    parser.add_argument("--ckpt_name", type=str, default="cifar10_model.pkl")
    parser.add_argument("--skip_model_save", action="store_true")
    
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--num_text_clusters", type=int, default=100)
    parser.add_argument("--agg_type", type=str, default='mean', choices=["mean","sum","median","max"])
    parser.add_argument("--dist_method", type=str, default="deep_knn", 
                        choices=["nn_pairwise", "nn_ot", "single_pairwise", "deep_knn","nn_pairwise_multimodal"])
    parser.add_argument("--deep_knn_thres", type=float, default=0.5)
    
    args = parser.parse_args()

    start_step = 0
    store_prefix = args.dataset
    args.base_folder = args.output_dir


    args.output_dir = os.path.join(
        args.output_dir, args.output_folder_name, args.store_name
    )

    tb_logger = Logger(logdir=args.output_dir, flush_secs=2)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print("Args:")
    for k, v in sorted(vars(args).items()):
        print("\t{}: {}".format(k, v))

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # NB: We vary random seeds along with seed for data split here (same seed).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.multiprocessing.set_sharing_strategy("file_system")

    num_workers = 2
    device = "cuda"

    cluster_kwargs = {}
    cluster_kwargs['n_clusters'] = args.num_text_clusters
    train_set, val_set, test_set = get_dataset(name=args.dataset, data_seed=args.seed,
                                                          noisy_labels=args.noise_labels,
                                                          percent_flips=args.percent_flips,
                                                         flip_type=args.flip_type,
                                                         return_combined_dataset=True,
                                                         cluster_text=True,
                                                         cluster_kwargs=cluster_kwargs)

        

    print("Batch size: ", args.batch_size)
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, num_workers=num_workers,
        shuffle=False
    )
    eval_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, num_workers=num_workers,
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, num_workers=num_workers,
        shuffle=False
    )
    

    print("Batch size: ", args.batch_size)
    img_base = get_img_base(args.img_base_name)
    tokenizer = AutoTokenizer.from_pretrained(args.text_base_name)
    algorithm = algorithm_class_from_scratch(
        args.algorithm, text_base_name=args.text_base_name, img_base=img_base
    )
    
    
    algorithm.to(device)
    algorithm.eval()

    pred_splits=["train", "test", "val"]
    if args.val_only:
        pred_splits=["test", "val"]

    
    
    label_flips = []
    dists = []
    bs = args.batch_size
    algorithm.eval()
    start_time = time.time()

    MMNB_obj = MMNB(algorithm, img_output_size=512, txt_output_size=512,
            tokenizer=tokenizer, label_to_word=np.zeros((MAX_SIZE, 1),dtype=str)[:,0],
                    device=device, agg_type=args.agg_type, dist_type=args.dist_type)
    MMNB_obj.fit(train_loader)

    all_label_flips=[]
    data_split_list=[]
    len_splits = []
    
    for phase in pred_splits:
        if phase == 'train':
            dataloader = train_loader
        elif phase == 'val':
            dataloader = eval_loader
        else:
            dataloader = test_loader


        i_ite = 0
        start_idx = 0
        len_splits.append(len(dataloader))
        for idx, batch in enumerate(tqdm(dataloader)):
            
            i_ite += 1
            
            if args.noise_labels:
                pixel_values, clean_labels, noisy_labels = batch
                labels = noisy_labels
            else:
                pixel_values, clean_labels = batch
                labels = clean_labels
                
                
            if args.noise_labels:
                label_flips = np.array(clean_labels)==np.array(noisy_labels)
                label_flips = 1-label_flips

            elif not args.noise_labels:
                raise NotImplementedError
                
            all_label_flips.append(label_flips)
            data_split_list.append([phase]*len(label_flips))
            pixel_values = pixel_values.to(device)
            img_embeds = algorithm.encode_image(pixel_values)
            
            

            if args.dist_method == "deep_knn":
                dist_batch = MMNB_obj.detect_label_knn(pixel_values,
                                                    k=args.knn_k, 
                                                    y_input=labels,
                                                    input_type=phase,
                                                    start_idx=start_idx)
                start_idx+=args.batch_size
                
            
            dists.extend(dist_batch)                
    
    end_time = time.time()
    runtime = end_time - start_time
    np.save(os.path.join(args.output_dir, "runtime.npy"), np.array([runtime]))
    np.save(os.path.join(args.output_dir, "len_splits.npy"), np.array(len_splits))

    np.save(os.path.join(args.output_dir, "label_flips_all.npy"), np.array(all_label_flips))
    np.save(os.path.join(args.output_dir, "dists.npy"), np.array(dists))
    np.save(os.path.join(args.output_dir, "datasplit.npy"), np.array(data_split_list))

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
    
   
