"""Script to parse input arguments and run models"""

import argparse
import json
import os
import random
import sys
import numpy as np
import PIL
import pickle
from tqdm import tqdm
import time

import torch
import torch.utils.data
from torch.utils.data import DataLoader

from lib.models.utils import get_captioning_processer_model
from lib.datasets.utils import get_captioning_dataset

def collate_fn(examples):
    pixel_values = []
    noisy_labels = []
    clean_labels = []

    for example in examples:
        pixels, clean_label, noisy_label = example
        pixel_values.append(torch.tensor(pixels.pixel_values[0]))  # pixel_values is a list with one element
        noisy_labels.append(noisy_label)
        clean_labels.append(clean_label)

    pixel_values = torch.stack(pixel_values)
    return {"pixel_values": pixel_values,  "clean_labels": clean_labels, "noisy_labels": noisy_labels}

def get_captfilt_scores(model, text_processor, dataloader, device='cpu'):
    scores = []
    flip_labels = []
    model.to(device)

    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixel_values = batch['pixel_values'].to(device)
            text_labels = list(batch['clean_labels'])
            if args.noise_labels:
                text_labels = list(batch['noisy_labels'])

            encoding = text_processor(text=text_labels, truncation=True,
                                padding=True, 
                                # max_length=100, #change made June 2
                                return_tensors="pt")
            

            encoding = {k:v for k,v in encoding.items()}
            input_ids = encoding.pop("input_ids").to(device)
            attention_mask = encoding.pop("attention_mask").to(device)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values)
            current_scores = outputs.itm_score.cpu().data.numpy()
            print(current_scores.shape)
            scores.extend(current_scores.squeeze())
            original_caption=np.array(batch['clean_labels'])
            noised_caption=np.array(batch['noisy_labels'])
            curr_flips=original_caption!=noised_caption

            flip_labels.extend(curr_flips.tolist())

    return scores, flip_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CaptFilt baseline")
    parser.add_argument("--exp_name", type=str, default="captfile")

    # training
    parser.add_argument(
        "--dataset",
        type=str,
        default="flickr30k",
        choices=["flickr30k", "mscoco", "mmimdb","mimiccxr_caption"],
    )
    parser.add_argument("--model_base_name", type=str, default="Salesforce/blip-itm-base-coco")
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

    # noise
    parser.add_argument("--flip_type", type=str, default="cat")
    parser.add_argument("--noise_labels", action="store_true")
    parser.add_argument("--percent_flips", type=float, default=0.3)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--dist_type", type=str, default="cosine")
    parser.add_argument("--method", type=str, default="ours")


    # hparams
    parser.add_argument("--batch_size", type=int, default=16)
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


    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
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

    num_workers = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_processor, text_processor, captfile_model = get_captioning_processer_model(args.model_base_name)



    train_set, val_set, test_set = get_captioning_dataset(args.dataset, args.data_seed, args.percent_flips,
                                                         args.flip_type, data_transform=img_processor)

    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False
    )
    eval_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False
    )
    dataloaders_dict={}
    dataloaders_dict['train']=train_loader
    dataloaders_dict['val']=eval_loader
    dataloaders_dict['test']=test_loader

    captfile_model.to(device)

    full_loss_dict={}
    flip_labels_dict={}
    start_time = time.time()
    len_splits = []
    for phase in ['train','val','test']:
        scores, flip_labels = get_captfilt_scores(model=captfile_model, text_processor=text_processor,
                                                  dataloader=dataloaders_dict[phase], device=device)
        full_loss_dict[phase]=scores
        flip_labels_dict[phase]=flip_labels
        len_splits.append(len(scores))

    end_time = time.time()
    runtime = end_time - start_time
    np.save(os.path.join(args.output_dir, "runtime.npy"), np.array([runtime]))
    np.save(os.path.join(args.output_dir, "len_splits.npy"), np.array(len_splits))

    with open(os.path.join(args.output_dir, 'flip_labels.pkl'), 'wb') as f:
        pickle.dump(flip_labels_dict, f)

    with open(os.path.join(args.output_dir, 'full_loss_dict.pkl'), 'wb') as f:
        pickle.dump(full_loss_dict, f)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
