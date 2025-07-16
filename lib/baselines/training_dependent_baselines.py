"""Script to parse input arguments and run models"""

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
import torch
import torchvision
import torch.utils.data
import torch.optim as optim
import time
from sklearn.metrics import f1_score, accuracy_score
from tensorboard_logger import Logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lib.utils.utils import EarlyStopping
from lib.models.utils import get_img_base 
from lib.datasets.utils import get_dataset
from lib.models.downstream_models import SuperviseClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multimodal distance metric')
    parser.add_argument('--exp_name', type=str, default="pretrain")

    # training
    parser.add_argument('--dataset', type=str, default="imagenet", choices=['imagenet','cifar10','cifar100',
                                                                            'mimic-cxr','stanford_cars','mini_imagenet'])
    parser.add_argument('--algorithm', type=str, default="medclip", choices=['huggingface_clip','medclip','finetune'])
    parser.add_argument('--text_base_name', type=str, default="albert-base-v2")
    parser.add_argument('--img_base_name', type=str, default="clipvisionmodelvit")
    parser.add_argument('--output_folder_name', type=str, default='debug')
    # others
    parser.add_argument("--flip_type", type=str, default="real")
    parser.add_argument("--baseline_type", type=str, default="aum")
    parser.add_argument("--noise_labels", action='store_true')
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 for "default hparams")')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')

    # early stopping
    parser.add_argument('--use_es', action='store_true')
    parser.add_argument('--es_strategy', choices=['metric'], default='metric')
    parser.add_argument('--lower_is_better', action='store_false')
    parser.add_argument('--es_metric', type=str, default='overall:loss')
    parser.add_argument('--es_patience', type=int, default=5, help='Stop after this many checkpoints w/ no improvement')

    # hparams
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum',default=0.9, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument("--percent_flips", type=float, default=0.3)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--do_softmax', action='store_true')


    # checkpoints
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--skip_model_save', action='store_true')
    args = parser.parse_args()

    start_step = 0
    store_prefix = args.dataset
    args.store_name = f"{store_prefix}_{args.algorithm}_hparams{args.hparams_seed}_seed{args.seed}"
    args.base_folder = args.output_dir
    args.output_dir = os.path.join(args.output_dir, args.output_folder_name, args.store_name)
    tb_logger = Logger(logdir=args.output_dir, flush_secs=2)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))


    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.multiprocessing.set_sharing_strategy('file_system')

    num_workers=2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set, val_set, test_set = get_dataset(args.dataset, args.seed, noisy_labels=args.noise_labels, 
                                                   percent_flips=args.percent_flips, flip_type=args.flip_type, 
                                                   return_combined_dataset=True,classifier_data_transform=None)

    print('Batch size: ', args.batch_size)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    eval_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    args.num_class = np.nan
    if args.dataset == 'cifar100' or args.dataset == 'mini_imagenet':
        args.num_class = 100
    elif args.dataset == 'stanford_cars':
        args.num_class = 196
    elif args.dataset == 'cifar10':
        args.num_class = 10
    else:
        raise NotImplementedError

    img_base=get_img_base(args.img_base_name)
    tokenizer=AutoTokenizer.from_pretrained(args.text_base_name)
    algorithm = SuperviseClassifier(img_base, num_class=args.num_class, mode='multiclass')

    es_group = args.es_metric.split(':')[0]
    es_metric = args.es_metric.split(':')[1]
    es = EarlyStopping(
        patience=args.es_patience, lower_is_better=args.lower_is_better)
    best_model_path = os.path.join(args.output_dir, 'model.best.pkl')


    if args.pretrained:
        raise NotImplementedError


    algorithm.to(device)


    def save_checkpoint(save_dict, filename='model.pkl'):
        if args.skip_model_save:
            return
        filename = os.path.join(args.output_dir, filename)
        torch.save(save_dict, filename)

    last_results_keys = None
    optimizer = optim.SGD(
            algorithm.parameters(), lr=args.lr, momentum=args.momentum)
    n_steps=args.epochs
    bs=args.batch_size
    start_time = time.time()
    for step in range(start_step, n_steps):
        if args.use_es and es.early_stop:
            print(f"Early stopping at step {step} with best {args.es_metric}={es.best_score}.")
            break
        step_start_time = time.time()

        for phase in ['train', 'val', 'test']:
            if phase=='train':
                dataloader=train_loader
                    
            elif phase=='val':
                dataloader=eval_loader
                
            elif phase=='test':
                dataloader=test_loader
        
            algorithm.train()

            loss_val_iter=[]
            y_pred_prob_all=[]
            y_true_all=[]
            label_flips_all=[]
            for idx, batch in enumerate(dataloader):
                pixel_values, text_labels, noise_labels_curr = batch

                binary_indices = noise_labels_curr.data.numpy()==text_labels.data.numpy()
                binary_indices = 1-np.array(binary_indices,dtype=np.int8)
                text_labels = torch.tensor(text_labels)
                        
                text_labels = text_labels.to(device)
                pixel_values = pixel_values.to(device)
                noise_labels_curr = torch.tensor(noise_labels_curr).to(device)
                optimizer.zero_grad()

                algorithm.train()
                outputs = algorithm.forward(pixel_values=pixel_values, labels = noise_labels_curr, 
                    return_loss=True,project='huggingface_clip')
                
                # NB: Need to train on test split as well for AUM/Datamap
                loss=outputs['loss_value']
                loss.backward()
                optimizer.step()
                assert not np.isnan(outputs['loss_value'].cpu().data.numpy())
                
                # Evaluating and storing predictions
                algorithm.eval()
                outputs = algorithm.forward(pixel_values=pixel_values, labels = noise_labels_curr, 
                    return_loss=True,project='huggingface_clip')
                y_pred_prob=torch.nn.Softmax(dim=1)(outputs["logits"]).detach().data.cpu().numpy()
                if not args.do_softmax:
                    y_pred_prob=outputs["logits"].detach().data.cpu().numpy()
                loss_val_iter.append(accuracy_score(text_labels.cpu().data.numpy(), np.argmax(y_pred_prob, axis=1)))
                y_pred_prob_all.extend(y_pred_prob)
                label_flips_all.extend(binary_indices.tolist())
                y_true_all.extend(noise_labels_curr.cpu().data.numpy())

            # NB: Getting validation metrics for FULL data split set after one epoch
            y_true_all = np.array(y_true_all).squeeze()
            y_pred_prob_all = np.array(y_pred_prob_all).squeeze()
            label_flips_all = np.array(label_flips_all).squeeze()
            save_dict = {
                "args": vars(args),
                "label_flips": label_flips_all,
                "prediction": y_pred_prob_all,
                "true_class": y_true_all,
            }
            if phase!='train':
                save_checkpoint(save_dict, filename='{}_epoch_{}.pkl'.format(phase, step))
                
            if phase=='val':
                es_metric_val=np.mean(loss_val_iter)
                if args.use_es:
                    es(es_metric_val, step, save_dict, best_model_path)

    end_time = time.time()
    runtime = end_time - start_time
    np.save(os.path.join(args.output_dir, "runtime.npy"), np.array([runtime]))

    with open(os.path.join(args.base_folder, 'done'), 'w') as f:
        f.write('done')

