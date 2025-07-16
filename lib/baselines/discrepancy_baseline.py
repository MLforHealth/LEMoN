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

from lib.models.utils import algorithm_class_from_scratch
from lib.datasets.utils import get_dataset, cifar10_labels, cifar100_labels, mini_imagenet_labels, stanford_cars_labels
from lib.utils.utils import path_serial, Tee, normalize_vectors
from lib.metrics import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"
clf_datasets = ["cifar10", "cifar100", 'cifar10_full','cifar100_full', 'mini_imagenet', 'stanford_cars']

parser = argparse.ArgumentParser(description="Baseline from ``Emphasizing Complementary Samples for Non-literal Cross-modal Retrieval``")
parser.add_argument("--exp_name", type=str)
parser.add_argument('--output_dir', type = str, required = True)
parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", 'flickr30k', 'mscoco', 'mimiccxr_caption', 'mmimdb', 'cifar10_full','cifar100_full', 
                                                                        'mini_imagenet', 'stanford_cars', 'cc3m'])
parser.add_argument("--noise_type", type=str, default="real", choices=["real", "asymmetric", "symmetric", "random", "noun", "cat"])
parser.add_argument("--method", type=str, default="dis_x", choices=["dis_x", "dis_y", "div_x", "div_y"])
parser.add_argument("--noise_level", type=float, default = 0.4)
parser.add_argument("--clip_model", type=str, default="huggingface_clip", choices = ["huggingface_clip", 'biomed_clip'])
parser.add_argument('--knn_k', default = 5, type = int)
parser.add_argument('--batch_size', default = 128, type = int)
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--data_seed', default = 0, type = int)
parser.add_argument('--compr_dataset_size_limit', default = 50000, type = int)
parser.add_argument('--custom_cifar_prompt', default = None)
parser.add_argument('--debug', action = 'store_true')
parser.add_argument('--skip_train', action = 'store_true')
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
    
    if args.clip_model == 'biomed_clip':
        encodings = tokenizer(text_labels).to(device)
    else:
        encodings = tokenizer(
                text_labels, padding="max_length", truncation=True)
        input_ids = torch.tensor(encodings["input_ids"]).to(device)
        attention_mask = torch.tensor(encodings["attention_mask"]).to(device)

    with torch.no_grad():
        if args.clip_model == 'biomed_clip':
            emb_txt.append(algorithm.encode_text(encodings).detach().cpu())
        else:
            emb_txt.append(algorithm.encode_text(input_ids, attention_mask).detach().cpu())
        emb_img.append(algorithm.encode_image(pixel_values).detach().cpu())
    
emb_txt_tr = normalize_vectors(torch.concat(emb_txt))
emb_img_tr = normalize_vectors(torch.concat(emb_img))

index_txt = faiss.IndexFlatIP(emb_txt_tr.shape[1])
index_img = faiss.IndexFlatIP(emb_img_tr.shape[1])
dists_tr = 1 - (emb_txt_tr * emb_img_tr).sum(axis = 1)

index_txt.add(emb_txt_tr.numpy())
index_img.add(emb_img_tr.numpy())
tr_text_labels = np.array(tr_text_labels)

logs = []
if args.debug or args.skip_train:
    sets_iter = zip(['val', 'test'],  [val_set, test_set])
else:
    sets_iter = zip(['train', 'val', 'test'], [train_set, val_set, test_set])

# build cache of NNs for train set
if 'dis' in args.method:
    _, cache = index_txt.search(emb_txt_tr.numpy(), k+ 1)
    cache = cache.tolist()
    for i in range(len(cache)):
        cache[i] = [j for j in cache[i] if j != i]

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
        
        if args.clip_model == 'biomed_clip':
            encodings = tokenizer(noisy_text_labels_prompts).to(device)
        else:
            encodings = tokenizer(
                    noisy_text_labels_prompts, padding="max_length", truncation=True)
            
            input_ids = torch.tensor(encodings["input_ids"]).to(device)
            attention_mask = torch.tensor(encodings["attention_mask"]).to(device)

        with torch.no_grad():
            if args.clip_model == 'biomed_clip':
                text_embeds = normalize_vectors(algorithm.encode_text(encodings).detach().cpu())
            else:
                text_embeds = normalize_vectors(algorithm.encode_text(input_ids, attention_mask).detach().cpu())
            img_embeds = normalize_vectors(algorithm.encode_image(pixel_values).detach().cpu())

        # D_ns, I_ns = index_img.search(img_embeds.numpy(), k + (sname == 'train'))
        D_ms, I_ms = index_txt.search(text_embeds.numpy(), k+ (sname == 'train'))

        for i in range(len(img_embeds)):
            sample_idx = idx * bs + i
            img_embed = img_embeds[i, None]
            text_embed = text_embeds[i, None]
            D_m, I_m = D_ms[i], I_ms[i]

            if args.method == 'dis_y':
                second_nns = [l for j in I_m for l in cache[j]]
                V = 1 - emb_txt_tr[second_nns] @ (text_embed.T)
                score = V.sum()/len(second_nns)
            elif args.method == 'dis_x':
                second_nns = [l for j in I_m for l in cache[j]]
                V = 1 - emb_img_tr[second_nns] @ (img_embed.T)
                score = V.sum()/len(second_nns)
            elif args.method == 'div_y':
                U = 1 - emb_txt_tr[I_m] @ (emb_txt_tr[I_m].T) 
                score = U.sum()/k**2
            elif args.method == 'div_x':
                U = 1 - emb_img_tr[I_m] @ (emb_img_tr[I_m].T) 
                score = U.sum()/k**2
                
            logs.append({
                'sset': sname,
                'idx': sample_idx,
                'actual_label': real_labels[i].item() if torch.is_tensor(real_labels[i]) else real_labels[i],
                'actual_label_text': clean_text_labels[i],
                'noisy_label': noisy_labels[i],
                'noisy_label_text': noisy_text_labels[i],
                'is_mislabel': label_flips[i],
                'is_correct_label': 1 - label_flips[i],
                'pred_score': score.item()
            })

df = pd.DataFrame(logs)

if not args.skip_train:
    df.to_csv(out_dir/'scores.csv', index = False)
    
selection_results = {}
df_val = df.query('sset == "val"')
thress = utils.eval_metrics(df_val['is_mislabel'],
                            df_val[f'pred_score'],
                            prevalence = df.loc[df.sset == 'val', 'is_mislabel'].sum()/(df.sset == 'val').sum())

for sset in df.sset.unique(): # eval score on each set
    sub_df = df.loc[df.sset == sset]
    selection_results[sset] = utils.eval_metrics(sub_df['is_mislabel'],
                                                                sub_df[f'pred_score'],
                                                                prevalence = df.loc[df.sset == 'val', 'is_mislabel'].sum()/(df.sset == 'val').sum(),
                                                                fix_thress = thress)            
res = {
    'df': df,
    'agg_results': selection_results
}
    
pickle.dump(res, (out_dir/'res.pkl').open('wb'))

if args.debug:
    import IPython; IPython.embed()

with open(os.path.join(out_dir, 'done'), 'w') as f:
    f.write('done')
