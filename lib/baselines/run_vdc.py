import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from lib.datasets.utils import get_dataset
from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm
import argparse
import json
from lib.utils.utils import path_serial, Tee
from pathlib import Path
import random
import socket
import pickle
from lib.metrics import utils
import transformers
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from lib.vdc.vqg import vqg_batched
from lib.vdc.vae import get_vdc_ae_score
from datetime import datetime
from lib.datasets.utils import get_dataset, cifar10_labels, cifar100_labels, mini_imagenet_labels, stanford_cars_labels

os.environ["TOKENIZERS_PARALLELISM"] = "false"

clf_datasets = ["cifar10", "cifar100", 'mini_imagenet', 'stanford_cars']

parser = argparse.ArgumentParser(description="VDC")
parser.add_argument("--exp_name", type=str)
parser.add_argument('--output_dir', type = str, required = True)
parser.add_argument("--dataset", type=str, default="flickr30k", choices=['flickr30k', 'mscoco', 'mimiccxr_caption', 'mmimdb', 
                                                                         ] + clf_datasets)
parser.add_argument("--noise_type", type=str, default="random", choices=["real", "asymmetric", "symmetric", "random", "noun", "cat"])
parser.add_argument("--noise_level", type=float, default = 0.4)
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--data_seed', default = 0, type = int)
parser.add_argument('--debug', action = 'store_true')
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

assert torch.cuda.is_available()

train_set, val_set, test_set = get_dataset(hparams['dataset'], args.data_seed, noisy_labels = True, percent_flips=args.noise_level, 
                                           flip_type=args.noise_type, return_combined_dataset = True, skip_transform=True)

if args.dataset in clf_datasets:
    label_set = {
            'cifar10': cifar10_labels,
            'cifar100': cifar100_labels,
            'stanford_cars': stanford_cars_labels,
            'mini_imagenet': mini_imagenet_labels
    }[args.dataset]

    for sset in [train_set, val_set, test_set]:
        if args.dataset in ['mini_imagenet', 'stanford_cars']:
            sset.df['sentence'] = sset.df['label'].apply(lambda x: label_set[x])
            sset.df['is_mislabel'] = 1 - sset.df['is_clean']
        else:
            xs, ys, noisy_ys = [], [], []
            for i in sset:
                xs.append(i[0])
                ys.append(label_set[i[1]])
                noisy_ys.append(label_set[i[2]])

            df = pd.DataFrame({
                'x': xs,
                'gold_sentence': ys,
                'sentence': noisy_ys,
            })
            df['is_mislabel'] = (df['sentence'] != df['gold_sentence'])
            sset.df = df

if args.debug:
    val_set.df = val_set.df.iloc[:20]
    test_set.df = test_set.df.iloc[:20]

if args.dataset == 'mini_imagenet':
    val_set.df = val_set.df.sample(n = 10000, random_state = 42)
    test_set.df = test_set.df.sample(n = 10000, random_state = 42)

# load models
llm_model = 'meta-llama/Llama-3.1-8B-Instruct'

llm_pipeline = transformers.pipeline(
    "text-generation", model=llm_model, 
    tokenizer=llm_model,
    model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto",
)

vlm_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
vlm_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", torch_dtype=torch.float16).to('cuda')

common_questions = ['Describe the image in detail.',
                    'Provide a detailed description of the given image.']

cifar10_questions = pd.read_csv('./lib/vdc/cifar10_specific_InstructBLIP.csv')
cifar10_questions['label'] = cifar10_questions.label.apply(lambda x: cifar10_labels[x])
cifar10_questions = cifar10_questions.groupby('label').apply(lambda x: list(x['question'])).to_dict()

start_t = datetime.now()

logs = []
for (sset, df) in zip(['val', 'test'], [val_set.df, test_set.df]):
    if args.dataset in clf_datasets:
        if args.dataset == 'cifar10':
            specific_questions = [cifar10_questions[i] for i in df.sentence.values]
        else:
            specific_questions = vqg_batched(llm_pipeline, df.sentence.values, batch_size = 64, clf = True)
    else:
        specific_questions = vqg_batched(llm_pipeline, df.sentence.values, batch_size = 64, clf = False)

    for c, (idx, row) in enumerate(tqdm(df.iterrows())):        
        with torch.inference_mode():
            # VQA
            if 'path' in row:
                image = Image.open(row['path']).convert("RGB") 
            else:
                image = row['x']
            inputs = vlm_processor(
                images = [image] * (len(common_questions) + len(specific_questions[c])),
                text = common_questions + specific_questions[c],
                return_tensors = 'pt',
                padding = True
            ).to(device="cuda", dtype=torch.float16)

            outputs = vlm_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    min_length=1,
                    do_sample = False
            )
            generated_text = vlm_processor.batch_decode(outputs, skip_special_tokens=True)
            vqa_answers = [i.strip() for i in generated_text]

            # VAE
            score = get_vdc_ae_score(row['sentence'], vqa_answers[:len(common_questions)], vqa_answers[len(common_questions):], ['yes'] * len(specific_questions[c]), llm_pipeline)

        log_i = {
            'sset': sset,
            'path': row['path'] if 'path' in row else None,
            'idx': idx,
            'actual_label_text': row['gold_sentence'] if 'gold_sentence' in row else None,
            'noisy_label_text': row['sentence'],
            'is_mislabel': row['is_mislabel'],
            'questions': common_questions + specific_questions[c],
            'answers': vqa_answers,
            'score': score
        }

        logs.append(log_i)

end_t = datetime.now()
timedelta = (end_t - start_t).total_seconds()
n_samples = len(logs)
print(f"Finished {n_samples} samples in {timedelta} seconds; avg of {timedelta/n_samples}s per sample")

df_final = pd.DataFrame(logs)

selection_results = {}
df_val = df_final.query('sset == "val"')
thress = utils.eval_metrics(df_val['is_mislabel'],
                        df_val[f'score'],
                        prevalence = df_final.loc[df_final.sset == 'val', 'is_mislabel'].sum()/(df_final.sset == 'val').sum())

for sset in df_final.sset.unique(): # eval score on each set
    sub_df = df_final.loc[df_final.sset == sset]
    selection_results[sset] = utils.eval_metrics(sub_df['is_mislabel'],
                                    sub_df['score'],
                                    prevalence = df_final.loc[df_final.sset == 'val', 'is_mislabel'].sum()/(df_final.sset == 'val').sum(),
                                    fix_thress = thress)
    
df_final.to_csv(out_dir/f'scores.csv', index = False)
res = {
    'df': df_final,
    'agg_results': selection_results
}
pickle.dump(res, (out_dir/'res.pkl').open('wb'))

if args.debug:
    import IPython; IPython.embed()

with open(os.path.join(out_dir, 'done'), 'w') as f:
    f.write('done')