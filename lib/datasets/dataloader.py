import os
from pathlib import Path
from PIL import Image,ImageFile

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

# from lib.datasets import load_dataset


class NoisyCombinedDataset(Dataset):
    def __init__(self, dataset, noise_labels, transform=None):
        self.original_dataset = dataset
        self.transform = transform
        self.noise_labels = noise_labels

    def __getitem__(self, index):
        x, y = self.original_dataset[index]
        y_noise = self.noise_labels[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, y_noise

    def __len__(self):
        return len(self.noise_labels)

class NoisyCombinedMultiModalDataset(Dataset):
    def __init__(self, dataset, noise_labels, transform=None):
        self.original_dataset = dataset
        self.transform = transform
        self.noise_labels = noise_labels

    def __getitem__(self, index):
        x1, x2, y= self.original_dataset[index]
        y_noise = self.noise_labels[index]
        if self.transform is not None:
            x = self.transform(x)
        return x1, x2, y, y_noise

    def __len__(self):
        return len(self.noise_labels)


ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    SPLITS = {               # Default, subclasses may override
        'tr': 'train',
        'va': 'validate',
        'te': 'test'
    }
    EVAL_SPLITS = ['te']     # Default, subclasses may override

    def __init__(self, root, split, split_path, label_path, metadata, transform, multimodal_report_root='/data/healthy-ml/gobi1/data/mimic-cxr-reports/files/',multimodal=False,
                 patient_file_path='/data/healthy-ml/scratch/aparnab/MultimodalDiscordance/data/patients.csv.gz', args=None):
        df_patient = pd.read_csv(patient_file_path)
        df_patient = df_patient.drop_duplicates(subset='subject_id')
        
        df = pd.read_csv(metadata)
        df_split = pd.read_csv(split_path)
        df_label = pd.read_csv(label_path)[['subject_id','study_id','No Finding']]
        df_label.loc[df_label['No Finding'].isna(),'No Finding']=0
        
        # merging with split info
        df=df.merge(df_split, on=['subject_id','study_id','dicom_id'])
        
        # merging with labels
        df=df.merge(df_label, on=['subject_id','study_id'])
        df=df.merge(df_patient[['subject_id','gender']], on='subject_id')
        
        df = df[df["split"] == (self.SPLITS[split])]
        df["filename"] = df.apply(lambda row: 'p{}/'.format(str(row.subject_id)[:2])+'p{}/'.format(
            row.subject_id) + 's{}/'.format(row.study_id)+'{}.jpg'.format(row.dicom_id), axis=1)
       
        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = df["No Finding"].tolist()
        self.transform_ = transform
        self.multimodal = multimodal
        self.args = args

        if self.multimodal:
            df["reportfilename"] = df.apply(lambda row: os.path.join(multimodal_report_root,
                                                                     'p{}/'.format(str(row.subject_id)[:2])+'p{}/'.format(row.subject_id) + 's{}.txt'.format(row.study_id)), axis=1)
        self.df = df


    def __getitem__(self, index):
        i = self.idx[index]
        x = self.transform(self.x[i])            
        y = torch.tensor(self.y[i], dtype=torch.long)
        if self.multimodal:
            with open(self.df.iloc[i]['reportfilename'],'r') as f:
                report = f.read()
            return x, report.split('IMPRESSION')[-1], y
        
        if self.args.name == 'simifeat':
            return x, y, index
        return x, y

    def __len__(self):
        return len(self.idx)

class LargeScaleDataset(Dataset):
    def __init__(self, df, transform, dataset_name):
        self.df = df
        self.transform = transform
        self.dataset_name =  dataset_name
    
    def __len__(self):
        return len(self.df)

    def get_image(self, x):    
        return Image.open(x).convert("RGB")
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        img = self.get_image(path)

        noisy_label = row['label']
        real_label = noisy_label if row['is_clean'] else noisy_label - 1 # since we don't know the real label, always return one that's different from noisy

        return self.transform(img), real_label, noisy_label
    
class ImageTextDataset(Dataset):
    """Image text dataset."""

    def __init__(self, data, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.img_transform = transform
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_img=self.data[idx][0]
        sample_img = sample_img.resize((224,224), Image.LANCZOS)
        sample_text=self.data[idx][1]
        sample_label=self.data[idx][2]

        if self.transform:
            sample = self.img_transform(sample_img)

        return sample, sample_text, sample_label


class CaptioningDataset(Dataset):
    def __init__(self, df, transform, dataset_name, use_cluster):
        self.df = df
        self.transform = transform
        self.dataset_name =  dataset_name
        self.use_cluster = use_cluster
    
    def __len__(self):
        return len(self.df)

    def get_image(self, x):
        if self.dataset_name == 'mimiccxr_caption':
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-5] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
            if reduced_img_path.is_file():
                return Image.open(reduced_img_path).convert("RGB")       
        return Image.open(x).convert("RGB")
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']

        if self.use_cluster:
            noisy_label = row['sent_cluster']
            real_label = -1 if row['is_mislabel'] else row['sent_cluster']
        else:
            real_label = row['gold_sentence']
            noisy_label = row['sentence']

        img = self.get_image(path)
        return self.transform(img), real_label, noisy_label
    

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
    