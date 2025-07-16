import pdb
import os
import copy
from collections import defaultdict
import requests

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, CLIPModel, ViTForImageClassification, ViTModel
import numpy as np
import torchvision

from . import constants

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))



class HuggingfaceCLIPModel(CLIPModel):
    def __init__(self, config):
        '''args:
        config: huggingface config name
        '''
        super().__init__(config)

    def encode_text(self, input_ids=None, attention_mask=None):
        return self.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    def encode_image(self, pixel_values=None):
        return self.get_image_features(pixel_values=pixel_values)



class SuperviseClassifier(nn.Module):
    '''take MedCLIP model with linear heads for supervised classification on images.
    '''
    def __init__(self,
        vision_model,
        num_class=14,
        input_dim=768,
        mode=None,
        freeze=False,
        **kwargs) -> None:
        '''args:
        vision_model: the medclip vision model that encodes input images into embeddings.
        num_class: number of classes to predict
        input_dim: the embedding dim before the linear output layer
        mode: multilabel, multiclass, or binary
        '''
        super().__init__()
        self.model = vision_model
        self.num_class = num_class
        assert mode.lower() in ['multiclass','multilabel','binary']
        self.mode = mode.lower()
        if num_class > 2:
            if mode == 'multiclass':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()

            self.fc = nn.Linear(input_dim, num_class)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.fc = nn.Linear(input_dim, 1)
        set_parameter_requires_grad(self.model, freeze)

    def forward(self,
        pixel_values,
        labels=None,
        return_loss=True,
        project='clip_pooled',
        **kwargs,
        ):
        outputs = defaultdict()
        # take embeddings before the projection head
        if project == 'huggingface_clip':
            img_embeds = self.model(pixel_values).pooler_output
        else:
            img_embeds = self.model(pixel_values)

        if isinstance(self.model, ViTModel):
            img_embeds = img_embeds.last_hidden_state[:, 0, :]


        logits = self.fc(img_embeds)
        outputs['embedding'] = img_embeds
        outputs['logits'] = logits

        if labels is not None and return_loss:
            if self.num_class==2: labels = labels.view(-1,1).float()
            if self.mode == 'multiclass': labels = labels.flatten().long()
            if self.num_class==2:
                loss = self.loss_fn(logits, labels)
            else:
                loss = self.loss_fn(logits, labels)
            outputs['loss_value'] = loss
        return outputs

