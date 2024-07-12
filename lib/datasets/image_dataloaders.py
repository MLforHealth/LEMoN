import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """Image dataset."""

    def __init__(self, data, label_flips=None,transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.transform = transform
        self.label_flips=label_flips
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_img=self.data[idx][0]
        sample_img = sample_img.resize((224,224), Image.LANCZOS)
        sample_label=self.data[idx][1]

        if self.transform:
            sample = self.transform(sample_img)

        return sample, sample_label
