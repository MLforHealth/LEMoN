"""Script to parse input arguments and run models"""

import argparse
import collections
import json
import os
import random
from scipy.special import softmax
import sys
import time
import numpy as np
import pandas as pd
import PIL
import pickle
import time
import torch
import torchvision
import torch.utils.data
import torch.optim as optim
from tensorboard_logger import Logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lib.models.utils import get_img_base, algorithm_class_from_scratch
from lib.datasets.utils import get_dataset, cifar10_labels, cifar100_labels, mini_imagenet_labels, stanford_cars_labels
from lib.metrics.distance_metrics import DistanceEvaluator

def get_label_to_word(dataset):
    if dataset=='cifar10':
        return cifar10_labels
    elif dataset=='cifar100':
        return cifar100_labels
    elif dataset=='mini_imagenet':
        return mini_imagenet_labels
    elif dataset=='stanford_cars':
        return stanford_cars_labels
    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal distance metric")
    parser.add_argument("--exp_name", type=str, default="pretrain")

    # training
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "cifar10", "cifar100", "mimic-cxr",
                 "stanford_cars","mini_imagenet"])
    parser.add_argument(
        "--algorithm",
        type=str,
        default="medclip",
        choices=["clip", "huggingface_clip", "finetune"],
    )
    parser.add_argument("--text_base_name", type=str, default="albert-base-v2")
    parser.add_argument(
        "--img_base_name",
        type=str,
        default="clipvisionmodel",
        choices=["clipvisionmodel", "clipvisionmodelvit"],
    )
    parser.add_argument("--output_folder_name", type=str, default="debug")

    # others
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--hparams_seed",
        type=int,
        default=0,
        help='Seed for random hparams (0 for "default hparams")',
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")

    # noise type
    parser.add_argument("--flip_type", type=str, default="random")
    # hparams
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--percent_flips", type=float, default=0.3)
    parser.add_argument("--noise_labels", action='store_true')
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--dist_type", type=str, default="cosine")
    parser.add_argument("--method", type=str, default="ours")

    # checkpoints
    parser.add_argument("--store_name", type=str, default=".")
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.multiprocessing.set_sharing_strategy("file_system")

    num_workers = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set, val_set, test_set = get_dataset(args.dataset, args.seed, noisy_labels=args.noise_labels,
                                                   percent_flips=args.percent_flips, flip_type=args.flip_type, 
                                                   return_combined_dataset=True)

    print("Batch size: ", args.batch_size)
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, num_workers=num_workers
    )
    eval_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, num_workers=num_workers
    )

    tokenizer = AutoTokenizer.from_pretrained(args.text_base_name)
    algorithm = algorithm_class_from_scratch(
        args.algorithm, text_base_name=args.text_base_name, img_base=None,
    )


    algorithm.to(device)
    algorithm.eval()
    
    start_time = time.time()
    n_steps = 1
    label_flips = []
    dists = []
    bs = args.batch_size
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        all_label_flips=[]
        data_split_list=[]
        for phase in ["train","val", "test"]:
            if phase == "train":
                dataloader = train_loader
            elif phase == "val":
                dataloader = eval_loader
            elif phase == "test":
                dataloader = test_loader
            else:
                raise NotImplementedError

            algorithm.eval()
            
            for idx, batch in enumerate(dataloader):
                pixel_values = batch[0]
                clean_labels = batch[1]
                label_set = get_label_to_word(args.dataset)
                clean_text_labels = label_set[clean_labels]
                text_label_for_pred = clean_text_labels

                #NB: We know nothing about label flips in train/val sets
                label_flips = [-1]*len(clean_labels)
                if args.noise_labels:
                    noisy_labels = batch[2]
                    noisy_text_labels = label_set[noisy_labels]
                    label_flips = np.array(noisy_labels)==np.array(clean_labels)
                    label_flips = 1-label_flips
                    text_label_for_pred = noisy_text_labels
                    noisy_prompt_labels = ['A photo of a '+ i for i in list(noisy_text_labels)]
                
                dataset_prompt_labels = ['A photo of a '+ i for i in label_set]

                    
                all_label_flips.append(label_flips)
                data_split_list.append([phase]*len(label_flips))

                # Getting the text representation for each label
                encodings = tokenizer(
                    dataset_prompt_labels, padding="max_length", truncation=True )
                input_ids = torch.tensor(encodings["input_ids"])
                attention_mask = torch.tensor(encodings["attention_mask"])
                input_ids = input_ids.to(device)
                pixel_values = pixel_values.to(device)
                attention_mask = attention_mask.to(device)

                text_embeds = algorithm.encode_text(input_ids, attention_mask)
                img_embeds = algorithm.encode_image(pixel_values)
                for img_ind in range(len(img_embeds)):
                    curr_noise_label = noisy_prompt_labels[img_ind]
                    img_repeats = img_embeds[img_ind]
                    img_repeats = img_repeats.repeat(len(text_embeds),1)
                    dist_obj = DistanceEvaluator(
                        y_true=None,
                        y_pred_proba=None,
                        dist=args.dist_type,
                        threshold=args.thresh,
                        y_pred_prob_epochs=None,
                        loss=None,
                        first_modality_embeddings=text_embeds,
                        second_modality_embeddings=img_repeats,
                    )
                    dist_embeds_per_img = dist_obj.our_metric()
                    dist_embeds_per_img = softmax(1-dist_embeds_per_img)
                    curr_img_pred_conf = dist_embeds_per_img[dataset_prompt_labels.index(curr_noise_label)]
                    dists.append(curr_img_pred_conf)
                

    end_time = time.time()
    runtime = end_time - start_time
    np.save(os.path.join(args.output_dir, "runtime.npy"), np.array([runtime]))
    np.save(os.path.join(args.output_dir, "label_flips.npy"), np.array(all_label_flips))
    np.save(os.path.join(args.output_dir, "dists.npy"), np.array(dists))
    np.save(os.path.join(args.output_dir, "datasplit.npy"), np.array(data_split_list))


    with open(os.path.join(args.base_folder, "done"), "w") as f:
        f.write("done")
