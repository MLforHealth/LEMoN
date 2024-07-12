import numpy as np
import torch
import torchvision
import torch
from torchvision import transforms
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pathlib import Path

from lib.datasets.dataloader import NoisyCombinedDataset, NoisyCombinedMultiModalDataset, CaptioningDataset
from lib.datasets.noise_captioning import random_noise_dict, noise_given_dict, calc_noise_by_integer_matching
from lib.datasets import clustering

PATHS = {
    'mscoco': '/data/healthy-ml/gobi1/data/mscoco/coco',
    'flickr30k': '/data/healthy-ml/gobi1/data/flickr30k/flickr30k_images',
    'mimic_cxr_reports': '/data/healthy-ml/gobi1/data/mimic-cxr-reports/files/',
    'mmimdb': '/data/healthy-ml/gobi1/data/MM-IMDB/mmimdb'
}

cifar10_labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
cifar10_labels = np.array(cifar10_labels)

cifar100_labels = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

cifar100_labels = np.array(cifar100_labels)

mimiccxr_labels = [
    "No finding",
    "Clinical finding"
]
mimiccxr_labels = np.array(mimiccxr_labels)

class_num_dict={}
class_num_dict['cifar10']=10
class_num_dict['cifar100']=100
class_num_dict['cifar10_full']=10
class_num_dict['cifar100_full']=100
class_num_dict['mimiccxr']=2

IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


generic_transform = transform = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ]
)

def add_noisy_labels(dataset, noise_type, noise_prop, data_seed=1, y_true=None):
    if noise_type == 'real':
        if dataset=='cifar10':
            noise_label = torch.load('./data/CIFAR-10_human.pt')['worse_label']
        elif dataset=='cifar100':
            noise_label = torch.load('./data/CIFAR-100_human.pt')['noisy_label']
        elif dataset=='mimiccxr':
            raise NotImplementedError
    else:
        assert y_true is not None
        assert noise_prop<1 and noise_prop>=0
        y_true = np.array(y_true)

        if noise_type == 'symmetric':
            noise_label, _ = noisify_multiclass_symmetric(y_true, noise_prop, 
                                                          random_state=data_seed,
                                                          nb_classes=class_num_dict[dataset])
        elif noise_type == 'asymmetric':
            noise_label, _ = noisify_pairflip(y_true, noise_prop, 
                                                          random_state=data_seed,
                                                          nb_classes=class_num_dict[dataset])
        else:
            raise NotImplementedError        
    return noise_label


# Adding scripts from simifeat https://github.com/UCSC-REAL/SimiFeat/tree/main
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print np.max(y), P.shape[0]
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print m
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    else:
        actual_noise = 0.

    return y_train, actual_noise


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    else:
        actual_noise = 0.

    return y_train, actual_noise

def get_captioning_dataset(name, data_seed, percent_flips, flip_type, 
                           data_transform=None,
                           cluster = False, cluster_kwargs = {}):
    '''
    flip_type: one of ['random', 'noun', 'cat']
    '''
    assert 0 <= percent_flips <= 1
    df = pd.read_pickle(Path(PATHS[name])/'multimodal_mislabel_split.pkl')
    if 'restval' in df.split:
        df.loc[df.split == 'restval', 'split'] = 'train'

    if name == 'mscoco':
        df['path'] = df.apply(lambda x: Path(PATHS['mscoco'])/x['filepath']/x['filename'], axis = 1)
    elif name == 'flickr30k':
        df['path'] = df.apply(lambda x: Path(PATHS['flickr30k'])/'flickr30k_images'/x['filename'], axis = 1)
    elif name == 'mimiccxr_caption':
        df['path'] = df.apply(lambda x: Path(PATHS['mimiccxr_caption'])/'files'/('p{}/'.format(str(x.subject_id)[:2])+'p{}/'.format(
            x.subject_id) + 's{}/'.format(x.study_id)+'{}.jpg'.format(x.dicom_id)), axis = 1)
        df = df[df.sentence.str.len() > 0] # 36 samples with no FINDINGs or IMPRESSIONS secion
    elif name == 'mmimdb':
        pass
    
    dfs = {}
    for split in ['train', 'val', 'test']:
        df_split = df.query(f'split == "{split}"')    
        if flip_type == 'random':
            noise_dict = random_noise_dict(len(df_split), percent_flips, data_seed)
        elif flip_type == 'noun':
            assert name in ['mscoco', 'flickr30k', 'mmimdb']
            noise_dict = calc_noise_by_integer_matching(df_split['nouns_int'].values, percent_flips, data_seed)
        elif flip_type == 'cat':
            assert name in ['mscoco', 'mimiccxr_caption', 'mmimdb']
            noise_dict = calc_noise_by_integer_matching(df_split['cat_labels'].values, percent_flips, data_seed)
        else:
            raise NotImplementedError(flip_type)
        dfs[split] = noise_given_dict(df_split, noise_dict)

    if cluster:
        km, train_clusters = clustering.cluster_caption_text(text_list = list(dfs['train']['sentence'].values), **cluster_kwargs)
        dfs['train']['sent_cluster'] =  train_clusters
        dfs['val']['sent_cluster'] = km.predict(list(dfs['val']['sentence'].values))
        dfs['test']['sent_cluster'] = km.predict(list(dfs['test']['sentence'].values))
    
    if data_transform is None:
        data_transform = generic_transform
    return (CaptioningDataset(dfs['train'], data_transform, name, cluster), 
            CaptioningDataset(dfs['val'], data_transform, name, cluster), 
            CaptioningDataset(dfs['test'], data_transform, name, cluster)
    )

def get_dataset(name, data_seed, noisy_labels=False, percent_flips=0.40,
                flip_type='real', multimodal=False,
                return_combined_dataset=True, cluster_text = False, cluster_kwargs= {'n_clusters': 100},
                caption_data_transform=None,
                return_indices=False):

    if name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        
        if noisy_labels:
            noise_labels = add_noisy_labels(dataset=name, noise_type=flip_type, noise_prop=percent_flips, 
                                            data_seed=data_seed, y_true=trainset.targets)


    elif name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        if noisy_labels:
            noise_labels = add_noisy_labels(dataset=name, noise_type=flip_type, noise_prop=percent_flips, 
                                            data_seed=data_seed, y_true=trainset.targets)
            
    elif name in ['cifar10_full', 'cifar100_full']:
        cls = torchvision.datasets.CIFAR10 if name == 'cifar10_full' else torchvision.datasets.CIFAR100
        trainset = cls(
            root="./data", train=True, download=True, transform=transform
        )
        testset = cls(
            root="./data", train=False, download=True, transform=transform
        )

        if noisy_labels:
            noise_labels_tr = add_noisy_labels(dataset=name, noise_type=flip_type, noise_prop=percent_flips, 
                                            data_seed=data_seed, y_true=trainset.targets)        
            noise_labels_te = add_noisy_labels(dataset=name, noise_type=flip_type, noise_prop=percent_flips, 
                                            data_seed=data_seed, y_true=testset.targets)
            
        train_indices, val_indices = train_test_split(np.arange(len(trainset)),test_size=0.2, random_state=data_seed)
        train_set = torch.utils.data.Subset(trainset, train_indices)
        val_set = torch.utils.data.Subset(trainset, val_indices)

        return NoisyCombinedDataset(train_set, noise_labels_tr[train_indices]), NoisyCombinedDataset(val_set, noise_labels_tr[val_indices]), NoisyCombinedDataset(testset, noise_labels_te)
    elif name in ['mscoco', 'flickr30k', 'mimiccxr_caption', 'mmimdb']:
        assert return_combined_dataset
        cluster_kwargs['clip_model'] = 'biomed_clip' if name == 'mimiccxr_caption' else 'huggingface_clip'
        return get_captioning_dataset(name, data_seed, percent_flips, flip_type, 
                                      data_transform=caption_data_transform,
                                      cluster = cluster_text, 
                                      cluster_kwargs=cluster_kwargs,
                                      )
    else:
        raise NotImplementedError
    
    # NB: Data split is : 80% training, 10% validation, 10% testing
    train_indices, val_indices = train_test_split(np.arange(len(trainset)),test_size=0.2, random_state=data_seed)
    val_indices, test_indices = train_test_split(val_indices,test_size=0.5, random_state=data_seed)

    if noisy_labels and return_indices: # used for SimiFeat
        return train_indices, val_indices, test_indices, noise_labels
    
    train_set = torch.utils.data.Subset(trainset, train_indices)
    val_set = torch.utils.data.Subset(trainset, val_indices)
    test_set = torch.utils.data.Subset(trainset, test_indices)

    if noisy_labels:
        if return_combined_dataset:
            return NoisyCombinedDataset(train_set, noise_labels[train_indices]), NoisyCombinedDataset(val_set, noise_labels[val_indices]), NoisyCombinedDataset(test_set, noise_labels[test_indices])
        else:
            return train_set, val_set, test_set, noise_labels[train_indices], noise_labels[val_indices], noise_labels[test_indices]
    elif return_combined_dataset:
        if multimodal:
                return NoisyCombinedMultiModalDataset(train_set, noise_labels[train_indices]), NoisyCombinedMultiModalDataset(val_set, noise_labels[val_indices]), NoisyCombinedMultiModalDataset(test_set, noise_labels[test_indices])
        return NoisyCombinedDataset(train_set, noise_labels[train_indices]), NoisyCombinedDataset(val_set, noise_labels[val_indices]), NoisyCombinedDataset(test_set, noise_labels[test_indices])
    
    else:
        return train_set, val_set, test_set