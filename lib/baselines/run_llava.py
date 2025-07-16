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
from lib.utils import path_serial, Tee, normalize_vectors
from lib.metrics.utils import binary_metrics, prob_metrics
from pathlib import Path
import random
import socket
import pickle
from lib.metrics import utils
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Multimodal kNN")
parser.add_argument("--exp_name", type=str)
parser.add_argument('--output_dir', type = str, required = True)
parser.add_argument("--dataset", type=str, default="flickr30k", choices=['flickr30k', 'mscoco', 'mimiccxr_caption', 'mmimdb'])
parser.add_argument("--noise_type", type=str, default="random", choices=["random", "noun", "cat"])
parser.add_argument("--noise_level", type=float, default = 0.4)
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--data_seed', default = 0, type = int)
parser.add_argument('--get_expl', action = 'store_true')
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
                                           flip_type=args.noise_type, return_combined_dataset = True)

if args.debug:
    val_set.df = val_set.df.iloc[:20]
    test_set.df = test_set.df.iloc[:20]

model_path = "liuhaotian/llava-v1.6-vicuna-13b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
model.eval()

if args.get_expl:
    query = 'The proposed caption for this image is "%s". Is this caption correct? Explain your answer.'
else:
    query = 'The proposed caption for this image is "%s". Is this caption correct? Only answer with "Yes" or "No".'

image_token_se = DEFAULT_IMAGE_TOKEN

def load_images_batched(image_files, batch_size = 256):
    processed_images = []
    image_sizes = []
    for i in range(0, len(image_files), batch_size):
        batch_images = [Image.open(file).convert("RGB") for file in image_files[i:i + batch_size]]
        processed_batch_images = process_images(
            batch_images,
            image_processor,
            model.config
        )
        processed_images.extend(processed_batch_images)
        image_sizes.extend([img.size for img in batch_images])
    return processed_images, image_sizes

def parse_output(st):
    if st is not None and st.lower().strip().startswith('no'):
        return 1
    return 0

logs = []
for (sset, df) in zip(['val', 'test'], [val_set.df, test_set.df]):
# for (sset, df) in zip(['test'], [test_set.df]):
    images_tensors, image_sizes = load_images_batched(df.path.values)

    for c, (idx, row) in enumerate(tqdm(df.iterrows())):
        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], image_token_se + "\n" + (query % row['sentence']))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensors[c].to(model.device, dtype=torch.float16),
                image_sizes=image_sizes[c],
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
                return_dict_in_generate=True, 
                output_scores=True
            )
            
        output_str = tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)[0].strip()

        transition_scores = model.compute_transition_scores(
            output_ids.sequences, output_ids.scores, normalize_logits=True
        ).cpu().numpy()[0] # equal to length of output string 

        if len(transition_scores) > 1 and not args.get_expl:
            output_prob = np.exp(transition_scores)[1] # skips first special token
        else:
            output_prob = 0.5
        
        log_i = {
            'sset': sset,
            'path': row['path'],
            'idx': idx,
            'actual_label_text': row['gold_sentence'],
            'noisy_label_text': row['sentence'],
            'is_mislabel': row['is_mislabel'],
            'raw_output': output_str
        }
        if not args.get_expl:
            log_i['pred'] = parse_output(output_str)

            if log_i['pred'] == 1: # output is "No" to "Is it correct"
                log_i['score'] = output_prob 
            else: # output is "Yes"
                log_i['score'] = 1 - output_prob            

        logs.append(log_i)

df_final = pd.DataFrame(logs)

selection_results = {}
if not args.get_expl:
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