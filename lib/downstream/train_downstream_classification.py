import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import random
import time
import logging
import torch
import glob
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import argparse
import PIL
import hashlib
import copy
import gc
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor

from lib.datasets.utils import get_dataset
from lib.metrics.utils import get_group_metrics, get_metrics
from lib.models.utils import get_img_base
from lib.models.downstream_models import SuperviseClassifier
from lib.utils.utils import NumpyEncoder

def get_trained_models(args, model=None, best_wts=None):
    if best_wts is not None:
        model.load_state_dict(best_wts)
        return model
    else:
        raise NotImplementedError('No model to load!')
    
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
    return pixel_values, torch.tensor(clean_labels).type(torch.LongTensor), torch.tensor(noisy_labels).type(torch.LongTensor)


def run_epoch(args, i, num_epochs, model, data_loader, optimizer, is_training, is_clean=True, weight=None, model_count=None,
              pred_split='val',scheduler=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_time = time.time()
    if is_training:
        model.train()
        loss_all=0
        b_idx=0
        with torch.set_grad_enabled(True):
            correct, total = 0, 0
            for batch_index, val in enumerate(data_loader):
                pixel_values, clean_labels, noisy_labels = val[0].to(device), val[1].to(device), val[2].to(device)
                assert len(pixel_values) == len(clean_labels)
                labels = noisy_labels
                outputs = model(pixel_values, labels=labels)
                loss = outputs['loss_value']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_all+=loss.item()
                b_idx+=1

                if not batch_index % 50:
                    print ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' 
                        %(i+1, num_epochs, batch_index, 
                            len(data_loader), loss.detach().cpu().numpy()))
        logging.info(f'Epoch: {i+1}/{num_epochs} | Time elapsed: {((time.time() - start_time)/60)} min')
        scheduler.step()

        return loss_all/b_idx
    else:
        model.eval()
        pred_y, true_y = [], []
        clean_y, noisy_y = [], []
        correct, total = 0, 0
        loss = None
        with torch.no_grad():
            for batch_index, val in enumerate(data_loader):
                pixel_values, clean_labels, noisy_labels = val[0].to(device), val[1].to(device), val[2].to(device)
                if is_clean:
                    labels = clean_labels
                else:
                    labels = noisy_labels
                outputs = model(pixel_values)
                y_pred_prob = torch.nn.Softmax(dim=1)(outputs["logits"]).detach().data.cpu().numpy()
                correct += np.sum(np.argmax(y_pred_prob, axis=1) == np.array(labels.cpu()))
                total += clean_labels.size(0)
                accuracy = 100*correct/total
                pred_y.extend(np.argmax(y_pred_prob, axis=1))
                true_y.extend(labels.cpu())
                clean_y.extend(clean_labels.cpu())
                noisy_y.extend(noisy_labels.cpu())
        try:
            assert np.abs(accuracy-100*accuracy_score(true_y,pred_y))<1e-5
        except:
            raise NotImplementedError('Accuracy not matching')
        logging.info(f'Time elapsed: {((time.time() - start_time)/60)} min | Accuracy: {accuracy}')

        if args.dataset == 'cifar10c' and args.num_epochs == 0:
            np.save(os.path.join(args.output_folder_updated, f"{pred_split}_pred_y_{args.cifar10_corruption_name}_{model_count}.npy"), np.array(pred_y))
            np.save(os.path.join(args.output_folder_updated, f"{pred_split}_true_y_{args.cifar10_corruption_name}_{model_count}.npy"), np.array(true_y))
        else:
            np.save(os.path.join(args.output_folder_updated, f"{pred_split}_pred_y.npy"), np.array(pred_y))
            np.save(os.path.join(args.output_folder_updated, f"{pred_split}_true_y.npy"), np.array(true_y))
            np.save(os.path.join(args.output_folder_updated, f"{pred_split}_clean_y.npy"), np.array(clean_y))
            np.save(os.path.join(args.output_folder_updated, f"{pred_split}_noisy_y.npy"), np.array(noisy_y))

        return loss, accuracy

def run(args):
    img_base = get_img_base(args.img_base, use_pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier_data_transform = AutoImageProcessor.from_pretrained(args.processor_base, do_resize=True,
                                                                   do_center_crop=True, 
                                                                   do_normalize=True, 
                                                                   do_rescale=True)

    if args.algorithm == 'supervised':
        input_dim = 768
        model = SuperviseClassifier(img_base, num_class=args.n_classes, input_dim=input_dim ,mode='multiclass',
                                    freeze=args.freeze)

    train_set, val_set, test_set = get_dataset(args.dataset, args.hparams_seed, noisy_labels=True, flip_type=args.flip_type, return_combined_dataset=True, percent_flips=0.4,
                                                   cifar10_corruption_name=args.cifar10_corruption_name,
                                                   classifier_data_transform=classifier_data_transform)

    weight=None
    
    if args.use_dist:
        dists=np.load(f"{args.folder}/dists.npy", allow_pickle=True)
        threshold = np.percentile(dists, args.percentile)
        pred_flips = dists > threshold
        pred_true_label_idx_train = np.where(pred_flips == 0)[0]
        assert len(pred_flips) == len(train_set)
        train_set = Subset(train_set, pred_true_label_idx_train)

    num_epochs = args.num_epochs

    model = model.to(device)
    params_to_update = model.parameters()
    if args.freeze:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    optimizer = optim.AdamW(params_to_update, lr=args.learning_rate)
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=args.learning_rate/100)


    counter = 0
    best_acc = float('-inf')
    best_model_wts = None
    for i in range(num_epochs):
        train_loss = run_epoch(args, i, num_epochs, model, DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn), optimizer, is_training=True, is_clean=args.clean_training, weight=weight,
                               scheduler=cosine_scheduler)
        val_loss, acc = run_epoch(args, i, num_epochs, model, DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn), optimizer, is_training=False, is_clean=args.clean_training,
                                  pred_split='val')
        print(train_loss, val_loss)
        logging.info(f'Training loss: {train_loss} | Validation loss: {val_loss}')
        if acc > best_acc:
            print("Saving best model")
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model, f"{args.output_folder_updated}/ckpt.pth")
            best_acc = acc
            counter = 0
        else:
            counter += 1
        if counter > args.es:
            break

    model = get_trained_models(args, model, best_model_wts)
    logging.info('-'*30)
    print(model)

    if args.num_epochs == 0:
        for model_count, _ in enumerate(model):
            run_epoch(args, 0, num_epochs, model[model_count], DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                                                          collate_fn=collate_fn), optimizer, is_training=False, is_clean=True, 
                    model_count = model_count,
                    pred_split='test')
    else:
        run_epoch(args, 0, num_epochs, model, DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                                         collate_fn=collate_fn), optimizer, is_training=False, is_clean=True,
                  pred_split='test')

def compute_acc(args, model_count=0, true_labels_known=True):
    pred_y=np.load(f"{args.output_folder_updated}/test_pred_y.npy", allow_pickle=True)
    true_y = np.load(f"{args.output_folder_updated}/test_true_y.npy", allow_pickle=True)
    clean_y = np.load(f"{args.output_folder_updated}/test_clean_y.npy", allow_pickle=True)
    noisy_y = np.load(f"{args.output_folder_updated}/test_noisy_y.npy", allow_pickle=True)
    if not true_labels_known:
        pred_y = pred_y[noisy_y==clean_y]
        true_y = true_y[noisy_y==clean_y]
    group_stats = get_group_metrics(pred_y, true_y, true_y, args.dataset)
    avg, worst = get_metrics(pred_y, true_y, group_stats)

    return {
        "average_accuracy": avg,
        "worst_accuracy": worst,
        "group_metrics": group_stats
    }

def compute_robustness_cifar10c(args, model_count=0):
    pred_y=np.load(f"{args.output_folder_updated}/test_pred_y_{args.cifar10_corruption_name}_{model_count}.npy", allow_pickle=True)
    true_y = np.load(f"{args.output_folder_updated}/test_true_y_{args.cifar10_corruption_name}_{model_count}.npy", allow_pickle=True)

    group_stats = get_group_metrics(pred_y, true_y, true_y, args.dataset)

    avg, worst = get_metrics(pred_y, true_y, group_stats)

    return {
        "average_accuracy": avg,
        "worst_accuracy": worst,
        "group_metrics": group_stats
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('--algorithm', type=str, default="supervised", choices=['supervised', 'base','huggingface_clip','medclip','finetune'])
    parser.add_argument('--img_base', type=str, default='resnet50', choices=['resnet50', 'clipvisionmodel','clipvisionmodelvit', 'dla', 'vit-base-patch16-224',
                                                                             'vit-base-patch16-224-base',
                                                                             'vit-base-patch16-224-model',
                                                                             'vit-base-patch16-224-base-cp'])
    parser.add_argument('--processor_base', type=str, default="google/vit-base-patch16-224")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100",
                 "cifar10c",'mini_imagenet','stanford_cars'],
    )
    parser.add_argument('--hparam-id', type=str)
    parser.add_argument('--id', type=int)
    parser.add_argument('--method', type=str)
    parser.add_argument("--use_dist", action='store_true', help="whether the dists are stored or not")
    parser.add_argument("--multimodal", action='store_true', help="whether the dataset is multimodal or not")
    parser.add_argument("--unfilter", action='store_true')
    parser.add_argument("--clean_training", action='store_true')
    parser.add_argument("--subsample_training_deepknn", action='store_true')
    parser.add_argument("--freeze", action='store_true')
    parser.add_argument("--true_labels_known", action='store_true')
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--flip_type", type=str)
    parser.add_argument("--cifar10_corruption_name", type=str)
    parser.add_argument("--store_name", type=str, default=".")
    parser.add_argument(
        "--hparams_seed",
        type=int,
        default=0,
        help='Seed for random hparams (0 for "default hparams")'
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--percentile", type=float, default=95)
    parser.add_argument('--es', type=int, default=3, help="early stopping counter")
    # Configuration
    parser.add_argument('--num_epochs', type=int,default=15)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--exp_name', type=str, help = 'Experiment name for downstream tracking purposes')

    args = parser.parse_args()

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))


    num_classes = {
        "cifar10": 10,
        "cifar10c": 10,
        "cifar100": 100,
        "mini_imagenet":100,
        "stanford_cars": 196,
    }

    args.n_classes = num_classes[args.dataset]

    if args.unfilter:
        args.folder = os.path.join(
            "output", "unfilter", args.dataset, args.flip_type
        )
    else:
        args.folder = os.path.join(
            "output", args.method, args.dataset, args.flip_type
        )
    args.output_folder = os.path.join(
        args.folder, args.img_base
    )
    if args.use_dist:
        args.output_folder = os.path.join(
            args.output_folder, str(args.percentile)
        )
        
    args_str = json.dumps(vars(args), sort_keys=True, cls = NumpyEncoder)
    args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
    
    if args.num_epochs == 0:
        args.output_folder_updated = os.path.join(
            args.output_folder, "inference"
        )
    else:
        args.output_folder_updated = os.path.join(
            args.output_folder, args_hash
        )


    output_dir = Path(args.output_folder_updated)
    output_dir.mkdir(parents = True, exist_ok=True)

    logging.basicConfig(filename=f"{args.output_folder_updated}/{args.num_epochs}_epochs.log", filemode='w+', level=logging.INFO, force=True)
    logging.info(f"{args}")
    
    with open(os.path.join(args.output_folder_updated, f"args_downstream.json"), "w+") as f:
        json.dump(vars(args), f, indent=4)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run(args)
    
    if args.dataset == 'cifar10c' and args.num_epochs == 0:
        for model_count in range(3):
            results = compute_robustness_cifar10c(args, model_count)
            pickle.dump(results, (output_dir/'results_{}_{}.pkl'.format(args.cifar10_corruption_name,
                                                                        model_count)).open('wb'))

    else:
        results = compute_acc(args, true_labels_known=args.true_labels_known)
        pickle.dump(results, (output_dir/'results.pkl').open('wb'))

    logging.shutdown()
    
    with open(os.path.join(args.output_dir, "done"), "w+") as f:
        f.write("done")

