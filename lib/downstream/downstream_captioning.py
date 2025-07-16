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
from tqdm import tqdm
import time

import torch
import torchvision
import torch.utils.data
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tensorboard_logger import Logger
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import AutoTokenizer
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model


from lib.utils.utils import EarlyStopping
from lib.models.utils import get_captioning_processer_model
from lib.datasets.utils import get_captioning_dataset

def collate_fn(examples):
    pixel_values = []
    noisy_labels = []
    clean_labels = []

    for example in examples:
        pixels, clean_label, noisy_label = example
        pixel_values.append(torch.tensor(pixels.pixel_values[0]))  # Assuming pixel_values is a list with one element
        noisy_labels.append(noisy_label)
        clean_labels.append(clean_label)

    pixel_values = torch.stack(pixel_values)
    return {"pixel_values": pixel_values,  "clean_labels": clean_labels, "noisy_labels": noisy_labels}


def shifted_lm_loss(logits, labels, device, loss_fct=CrossEntropyLoss(reduction="none"), vocab_size=30522, num_img_tokens=197):
    assert logits.dim() == 3  # (batch_size, seq_len, vocab_size)
    assert labels.dim() == 2  # (batch_size, seq_len)
    
    if loss_fct is None:
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    # Shift for autoregressive prediction: predict next token
    shift_logits = logits[..., num_img_tokens:-1, :].contiguous()  # Remove last position
    shift_labels = labels[..., 1:].contiguous()      # Remove first position
    
    # Now logits[i] predicts labels[i+1]
    seq_len_logits = shift_logits.size(1)
    seq_len_labels = shift_labels.size(1)

    # Align sequence lengths (same logic as before)
    if seq_len_logits > seq_len_labels:
        pad_len = seq_len_logits - seq_len_labels
        pad_val = torch.full((shift_labels.size(0), pad_len), -100, dtype=torch.long, device=device)
        shift_labels = torch.cat((shift_labels, pad_val), dim=1)
    elif seq_len_logits < seq_len_labels:
        shift_labels = shift_labels[:, :seq_len_logits]

    # Reshape and compute loss
    shift_logits = shift_logits.transpose(1, 2)  # (B, vocab_size, seq_len)
    loss_per_token = loss_fct(shift_logits, shift_labels)
    mask = (shift_labels.view(labels.size(0), -1) != -100).float()
    loss_per_sample = (loss_per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return loss_per_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal distance metric")
    parser.add_argument("--exp_name", type=str, default="caption")

    # training
    parser.add_argument(
        "--dataset",
        type=str,
        default="flickr30k",
        choices=["flickr30k", "mscoco", "mmimdb","mimiccxr_caption"],
    )
    parser.add_argument("--model_base_name", type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--output_folder_name", type=str, default="./")

    # others
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--data_seed",
        type=int,
        default=0,
        help='Seed for data (0 for "default")',
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument("--tied_seed", action='store_true')


    # early stopping
    parser.add_argument("--flip_type", type=str, default="cat")
    parser.add_argument("--train_all", action="store_true")
    parser.add_argument("--noise_labels", action="store_true")
    parser.add_argument("--use_es", action="store_true")
    parser.add_argument("--es_strategy", choices=["metric"], default="metric")
    parser.add_argument("--lower_is_better", action="store_true")
    parser.add_argument("--es_metric", type=str, default="overall:loss")
    parser.add_argument(
        "--es_patience",
        type=int,
        default=5,
        help="Stop after this many checkpoints w/ no improvement",
    )

    # hparams
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--percent_flips", type=float, default=0.3)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--dist_type", type=str, default="cosine")
    parser.add_argument("--method", type=str, default="ours")
    parser.add_argument("--runscheduler", action="store_true")

    # hparams: text generation
    parser.add_argument("--prompt", type=str, default="NA")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--beam_es", action='store_true')
    parser.add_argument("--do_sampling", action='store_true')
    parser.add_argument("--do_text_generation", action='store_true')
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--percentile_filter", type=float, default=0.95)
    parser.add_argument("--score_col", type=str, default='pred_score')
    parser.add_argument("--split_col", type=str, default='sset')

    # hparams: LoRA approximation
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # hparams: subsetting
    parser.add_argument("--filter_data", action='store_true')
    parser.add_argument("--subset_csv_path", type=str, default="./know_val_labels_scores.csv")

    # checkpoints
    parser.add_argument("--resume", "-r", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--store_name", type=str, default=".")
    parser.add_argument("--ckpt_name", type=str, default="model.pkl")
    parser.add_argument(
        "--checkpoint_freq", type=int, default=1, help="Checkpoint every N steps"
    )
    parser.add_argument("--skip_model_save", action="store_true")
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

    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_processor, text_processor, algorithm = get_captioning_processer_model(args.model_base_name)


    if args.tied_seed: 
        args.data_seed=args.seed
    train_set, val_set, test_set = get_captioning_dataset(args.dataset, args.data_seed, args.percent_flips,
                                                         args.flip_type, data_transform=img_processor)


    # Performing LoRA for better finetuning
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj"]
    )

    algorithm = get_peft_model(algorithm, lora_config)


    # n_steps = args.steps
    checkpoint_freq = args.checkpoint_freq
    if args.filter_data:
        df_filter = pd.read_csv(args.subset_csv_path)
        df_filter = df_filter[df_filter[args.split_col]=='train']
        percentile_val = np.percentile(df_filter[args.score_col].values, args.percentile_filter)
        df_filter['is_correct_label'] = np.array(df_filter[args.score_col].values < percentile_val, dtype=np.int8)
        filter_indicators = df_filter['is_correct_label'].values
        filtered_indices=np.arange(df_filter.shape[0])[filter_indicators==1]
        train_set =  Subset(train_set, filtered_indices)
        try:
            assert np.abs((len(filtered_indices)/df_filter.shape[0])-args.percentile_filter) < 0.05
        except:
            print('Filtering:',len(filtered_indices)/df_filter.shape[0])


    print("Batch size: ", args.batch_size)
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, 
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn
    )
    train_loader_unshuffled = DataLoader(
        dataset=train_set, batch_size=args.batch_size, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    eval_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, num_workers=num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, num_workers=num_workers,
        collate_fn=collate_fn
    )

    train_minibatches_iterator = iter(train_loader)
    checkpoint_vals = collections.defaultdict(lambda: [])


    es = EarlyStopping(
        patience=args.es_patience, lower_is_better=args.lower_is_better)
    best_model_path = os.path.join(args.output_dir, 'model.best.pkl')


    optimizer = AdamW(algorithm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t_total = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
    num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    
    algorithm.to(device)

    label_flips = []
    dists = []
    bs = args.batch_size

    if args.train_all and args.use_es:
        raise NotImplementedError
    
    dataloaders_dict={}
    dataloaders_dict['train']=train_loader
    dataloaders_dict['train_unshuffled']=train_loader_unshuffled
    dataloaders_dict['val']=eval_loader
    dataloaders_dict['test']=test_loader


    if args.train_all:
        train_splits=['train','val','test']
        val_splits=['val','test']
    else:
        train_splits=['train']
        val_splits=['val']
            
    full_loss_dict={}
    full_loss_dict['train_unshuffled']=[]
    full_loss_dict['val']=[]
    full_loss_dict['test']=[]

    start_time = time.time()
    len_splits = []
    for step in range(start_step, args.epochs):
        step_start_time = time.time()
        algorithm.train()
        
        for split in train_splits:
            for batch in tqdm(dataloaders_dict[split]):
                # returned value is a dict with key pixel_values
                pixel_values = batch['pixel_values'].to(device)
                text_labels = list(batch['clean_labels'])
                assert len(pixel_values) == len(text_labels)
                if args.noise_labels:
                    text_labels = list(batch['noisy_labels'])

                encoding = text_processor(text=text_labels, truncation=True,
                                    padding=True, 
                                    max_length=100,
                                    return_tensors="pt")

                
                # remove batch dimension
                encoding = {k:v for k,v in encoding.items()}
                input_ids = encoding.pop("input_ids").to(device)
                attention_mask = encoding.pop("attention_mask").to(device)
                
                optimizer.zero_grad()
                outputs = algorithm(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                labels=input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                if args.runscheduler:
                    scheduler.step()
                
            
        algorithm.eval()

        with torch.no_grad():
            for split in val_splits:
                loss_val = []
                curr_loader_loss = []
                len_splits.append(len(dataloaders_dict[split]))
                for batch in tqdm(dataloaders_dict[split]):
                    pixel_values = batch['pixel_values'].to(device)
                    text_labels = list(batch['clean_labels'])
                    assert len(pixel_values) == len(text_labels)
                    if args.noise_labels:
                        text_labels = list(batch['noisy_labels'])

                    encoding = text_processor(text=text_labels, truncation=True,
                                        padding=True, max_length=100,
                                        return_tensors="pt")

                    
                    # remove batch dimension
                    encoding = {k:v for k,v in encoding.items()}
                    input_ids = encoding.pop("input_ids").to(device)
                    attention_mask = encoding.pop("attention_mask").to(device)
                    outputs = algorithm(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                labels=input_ids)
                    loss = outputs.loss
                    loss_val.append(loss.item())

                    if args.train_all:
                        full_loss = shifted_lm_loss(outputs.logits,
                        input_ids,device)
                        curr_loader_loss.extend(full_loss.cpu().data.numpy().squeeze().tolist())
                
                if args.use_es:
                    assert len(val_splits) == 1
                    assert val_splits[0] == "val"
                    print(np.mean(loss_val))
                    save_dict = {
                    "args": vars(args),
                    "model_dict": algorithm.state_dict()}
                    es(np.mean(loss_val), step, save_dict, best_model_path)
                full_loss_dict[split].append(curr_loader_loss)
    
    end_time = time.time()            
    if args.epochs>0 and args.use_es:
        best_state_dict = torch.load(best_model_path)
        algorithm.load_state_dict(best_state_dict['model_dict'])
        
    algorithm.eval()


    # Captioning

    generated_captions = {}
    flip_labels = {}

    
    with torch.no_grad():
        for phase in ["val","test"]:
            generated_captions[phase]=[]
            flip_labels[phase]=[]
            if phase == "val":
                dataloader = eval_loader
            elif phase == "test":
                dataloader = test_loader
            else:
                raise NotImplementedError
            split_flips=[]
            generated_captions_curr=[]
            datasplits_curr=[]
            for batch in tqdm(dataloader):
                pixel_values = batch['pixel_values'].to(device)
                text_labels = list(batch['clean_labels'])
                assert len(pixel_values) == len(text_labels)
                if args.noise_labels:
                    text_labels = list(batch['noisy_labels'])
                if args.do_text_generation:
                    if args.do_sampling:
                        generated_ids = algorithm.generate(pixel_values=pixel_values,
                                    max_length=args.max_length,
                                    top_k=args.top_k,
                                    temperature=args.temperature,
                                    do_sample=True
                                    )
                    else:
                        generated_ids = algorithm.generate(pixel_values=pixel_values,
                                    num_beams=args.num_beams,
                                    max_length=args.max_length,
                                    early_stopping=args.beam_es
                                    )

                    generated_caption = text_processor.batch_decode(generated_ids, skip_special_tokens=True)
                    generated_captions[phase].extend(list(generated_caption))

                original_caption=np.array(batch['clean_labels'])
                noised_caption=np.array(batch['noisy_labels'])
                curr_flips=original_caption!=noised_caption

                flip_labels[phase].extend(curr_flips.tolist())

    runtime = end_time - start_time
    np.save(os.path.join(args.output_dir, "runtime.npy"), np.array([runtime]))
    np.save(os.path.join(args.output_dir, "len_splits.npy"), np.array(len_splits))

    
    with open(os.path.join(args.output_dir, 'captions.pkl'), 'wb') as f:
        pickle.dump(generated_captions, f)

    with open(os.path.join(args.output_dir, 'flip_labels.pkl'), 'wb') as f:
        pickle.dump(flip_labels, f)

    if args.train_all:
        with open(os.path.join(args.output_dir, 'full_loss_dict.pkl'), 'wb') as f:
            pickle.dump(full_loss_dict, f)
         
    #NB: Deleting the model to save space
    if args.use_es:
        os.remove(best_model_path)


    with open(os.path.join(args.base_folder, 'done'), 'w') as f:
        f.write('done')

