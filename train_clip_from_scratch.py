# Adapted from https://github.com/rajpurkarlab/CheXzero/blob/main/run_train.py

# python train_clip_from_scratch.py --output_dir "/mnt/scratch-lids/scratch/haoran/results/MultimodalDiscordance/results/clip_scratch/mimic_cat_40" --noise_type cat
# python train_clip_from_scratch.py --output_dir "/mnt/scratch-lids/scratch/haoran/results/MultimodalDiscordance/results/clip_scratch/mimic_random_40" --noise_type random

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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.utils.utils import path_serial, Tee
from lib.datasets.utils import get_dataset
from lib.models.chexzero_clip import load_clip, tokenize

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Train CLIP on MIMIC-CXR")
parser.add_argument("--exp_name", type=str)
parser.add_argument('--output_dir', type = str, required = True)
parser.add_argument("--dataset", type=str, default="mimiccxr_caption", choices=['flickr30k', 'mscoco', 'mimiccxr_caption', 'mmimdb'])
parser.add_argument("--noise_type", type=str, default="cat", choices=["random", "noun", "cat"])
parser.add_argument("--noise_level", type=float, default = 0.4)
parser.add_argument('--batch_size', default = 64, type = int)
parser.add_argument('--epochs', default = 10, type = int)
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--data_seed', default = 0, type = int)
parser.add_argument('--debug', action = 'store_true')
parser.add_argument('--save_interval', type=int, default=2000)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--optimizer', type=str, default="sgd")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=1e-4)
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
device = 'cuda'

with open(out_dir/'args.json', 'w') as outfile:
    json.dump(vars(args), outfile, default=path_serial)

train_set, val_set, test_set = get_dataset(hparams['dataset'], args.data_seed, noisy_labels = True, percent_flips=args.noise_level, 
                                           flip_type=args.noise_type, return_combined_dataset = True)

loader = DataLoader(
        dataset=train_set, batch_size=hparams['batch_size'], num_workers=8, shuffle = True
)

model = load_clip(model_path=None).to(device)

criterion = nn.CrossEntropyLoss().cuda()
if args.optimizer == "adam": 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
elif args.optimizer == "sgd": 
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

total_batches = len(loader) * args.epochs

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(6) + " examples and " + str(epoch).zfill(3) + f" epochs: {loss:.3f}")
    
def save(model, path): 
    torch.save(model.state_dict(), path)

example_ct = 0  # number of examples seen
batch_ct = 0
report_freq = args.log_interval

for epoch in range(args.epochs):
    running_loss = 0.0 # running loss over batch
    for data in tqdm(loader):
        # get the images
        images = data[0].to(device)
        texts = tokenize(data[2], model).to(device)
        
        # perform step for a single batch
        logits_per_image, logits_per_text = model(images, texts)
        batch_size = images.shape[0]
        labels = torch.arange(batch_size).to(device)

        loss_img = criterion(logits_per_image, labels)
        loss_txt = criterion(logits_per_text, labels)
        loss = (loss_img + loss_txt)/2 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        example_ct +=  len(images)
        batch_ct += 1
        running_loss += loss.item()

        # Report metrics every `report_freq` batch
        if (batch_ct % report_freq) == 0:
            train_log(running_loss / report_freq, example_ct, epoch)
            running_loss = 0.0
        
        if (batch_ct % args.save_interval) == 0: 
            model_path = os.path.join(out_dir, "checkpoint_{batch_ct}.pt".format(
                batch_ct=str(batch_ct), 
            ))
            print("Saved checkpoint to: ", model_path)
            save(model, model_path)

with open(os.path.join(out_dir, 'done'), 'w') as f:
    f.write('done')
