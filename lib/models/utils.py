import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torchvision.models as models
from transformers import ViTForImageClassification

from open_clip import create_model_from_pretrained, get_tokenizer
from transformers import AutoTokenizer

from transformers import AutoTokenizer, CLIPModel
from transformers import BlipForImageTextRetrieval,CLIPVisionModel, AutoProcessor, AutoImageProcessor, Blip2ForConditionalGeneration, AutoModelForCausalLM
from torchvision import transforms
from PIL import Image
from lib.models.chexzero_clip import load_clip, tokenize
from lib.models.downstream_models import HuggingfaceCLIPModel
import clip

our_model_paths = {
    'random': '/mnt/scratch-lids/scratch/haoran/results/MultimodalDiscordance/results/clip_scratch/mimic_random_40/checkpoint_30000.pt',
    'cat': '/mnt/scratch-lids/scratch/haoran/results/MultimodalDiscordance/results/clip_scratch/mimic_cat_40/checkpoint_30000.pt',
    'chexzero': '/data/healthy-ml/scratch/haoran/repos/CheXzero/checkpoints/best_64_5e-05_original_22000_0.864.pt',
    'cc3m_clip_from_scratch': '/data/healthy-ml/scratch/haoran/results/MultimodalDiscordanceNew/results/cc3m_clip_scratch/faf3fa4b159a6c7da61bb301d6ca3f42/checkpoint_400000.pt'
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
    if name=='resnet50':
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
    if name=='huggingface_clip':
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
    elif name.startswith('cc3m_clip_from_scratch'):
        model = load_clip(model_path=our_model_paths[name], context_length=77)
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

