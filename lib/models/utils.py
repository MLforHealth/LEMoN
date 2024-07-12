import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torchvision.models as models
from transformers import ViTForImageClassification

from open_clip import create_model_from_pretrained, get_tokenizer
from transformers import AutoTokenizer

from .medclip_model import MedCLIPModel,MedCLIPVisionModel, MedCLIPVisionModelViT, HuggingfaceCLIPModel, SuperviseClassifier
from transformers import BlipForImageTextRetrieval,CLIPVisionModel, AutoProcessor, AutoImageProcessor, Blip2ForConditionalGeneration, AutoModelForCausalLM
from torchvision import transforms
from PIL import Image
from lib.models.chexzero_clip import load_clip, tokenize
import clip

our_model_paths = {
    'random': '/mnt/scratch-lids/scratch/haoran/results/MultimodalDiscordance/results/clip_scratch/mimic_random_40/checkpoint_30000.pt',
    'cat': '/mnt/scratch-lids/scratch/haoran/results/MultimodalDiscordance/results/clip_scratch/mimic_cat_40/checkpoint_30000.pt',
    'chexzero': '/data/healthy-ml/scratch/haoran/repos/CheXzero/checkpoints/best_64_5e-05_original_22000_0.864.pt'
}

def get_captioning_processer_model(model_base_name):
    if 'blip2' in model_base_name:
        processor = AutoProcessor.from_pretrained(model_base_name, padding="max_length",
        truncation=True, return_tensors="pt")
        model = Blip2ForConditionalGeneration.from_pretrained(model_base_name)
        return processor, processor, model
    elif 'git' in model_base_name:
        img_processor = AutoImageProcessor.from_pretrained(model_base_name, padding="max_length",
        truncation=True, return_tensors="pt")
        text_processor = AutoProcessor.from_pretrained(model_base_name, padding="max_length",
        truncation=True, return_tensors="pt")
        model = AutoModelForCausalLM.from_pretrained(model_base_name)
        return img_processor, text_processor, model
    elif 'itm' in model_base_name:
        processor = AutoProcessor.from_pretrained(model_base_name, padding="max_length",
        truncation=True, return_tensors="pt")
        model = BlipForImageTextRetrieval.from_pretrained(model_base_name)
        return processor, processor, model
    else:
        raise NotImplementedError
        

def get_img_base(name, embed_dim=768, use_pretrained=False):
    if name=='clipvisionmodel':
        img_base=MedCLIPVisionModel
    elif name=='clipvisionmodelvit':
        img_base=MedCLIPVisionModelViT
    elif name=='resnet50':
        img_base=models.resnet50(pretrained=use_pretrained)
        num_ftrs = img_base.fc.in_features
        # NB: num_classes is 768 here!!
        img_base.fc = nn.Linear(num_ftrs, embed_dim)
    elif name=='vit-base-patch16-224':
        img_base= ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    elif name=='openai/clip-vit-base-patch32':
        img_base= CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
    else:
        raise NotImplementedError
    return img_base
    
def algorithm_class_from_scratch(name,text_base_name,img_base, return_tokenizer = False):
    if name=='medclip':
        #img_base=get_img_base(img_base)
        print(img_base)
        return MedCLIPModel(bert_type=text_base_name,vision_cls=img_base)
    elif name=='huggingface_clip':
        tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        model = HuggingfaceCLIPModel.from_pretrained(text_base_name)
        if return_tokenizer:
            return model, tokenizer
        else:
            return model
    elif name=='biomed_clip':
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        if return_tokenizer:
            return model, tokenizer
        else:
            return model
    elif name.startswith('mimic_clip_from_scratch'):
        typ = name.split('_')[-1]
        model = load_clip(model_path=our_model_paths[typ])
        tokenizer = lambda x: tokenize(x, model)
        if return_tokenizer:
            return model, tokenizer
        else:
            return model
    elif name == 'chexzero':
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        model, transform = clip.load("ViT-B/32", device='cuda', jit=False)
        model.load_state_dict(torch.load(our_model_paths['chexzero']))
        tokenizer = lambda x: tokenize(x, model)
        model.float()
        if return_tokenizer:
            return model, tokenizer
        else:
            return model
    else:
        raise NotImplementedError

def train_model_across_epochs(model, train_loader, epochs=20, criterion=torch.nn.CrossEntropyLoss()):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            img_inp = batch['images'].to(device)
            tab_inp = batch['tabular_pts'].to(device)
            loss = model(
                img_inp, tab_inp)

            loss.backward()
            optimizer.step()
    return model

def get_val_loss(model, val_loader, criterion=torch.nn.CrossEntropyLoss()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss_iter=[]
    for batch in val_loader:
        img_inp = batch['images'].to(device)
        tab_inp = batch['tabular_pts'].to(device)
        loss = model(
            img_inp, tab_inp)
        loss_iter.append(loss.item())
    loss_iter=np.array(loss_iter)
    loss_iter=loss_iter.squeeze()
    return np.mean(loss_iter)