import numpy as np
import torch
import torchvision
import torch
from torchvision import transforms
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pathlib import Path

from lib.datasets.dataloader import NoisyCombinedDataset, NoisyCombinedMultiModalDataset, CaptioningDataset, LargeScaleDataset
from lib.datasets.noise_captioning import random_noise_dict, noise_given_dict, calc_noise_by_integer_matching
from lib.datasets import clustering

PATHS = {
    'mscoco': '/data/healthy-ml/gobi1/data/mscoco/coco',
    'flickr30k': '/data/healthy-ml/gobi1/data/flickr30k/flickr30k_images',
    'mimic_cxr_reports': '/data/healthy-ml/gobi1/data/mimic-cxr-reports/files/',
    'mmimdb': '/data/healthy-ml/gobi1/data/MM-IMDB/mmimdb',
    'mimiccxr_caption': '/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG',
    'mini_imagenet': '/data/healthy-ml/gobi1/data/ImageNetRed/dataset_no_images/mini-imagenet',
    'stanford_cars': '/data/healthy-ml/gobi1/data/ImageNetRed/dataset_no_images/stanford_cars',
    'cc3m': '/data/healthy-ml/gobi1/data/cc3m-wds/cc3m-wds/'
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

mini_imagenet_labels = np.array(['triceratops', 'upright piano', 'Gordon setter', 'cocktail shaker', 'unicycle, monocycle', 'organ, pipe organ', 'Alaskan malamute', 'prayer rug', 'Newfoundland dog', 'tobacco shop', 'ladybug', 'combination lock', 'ashcan, trash can', 'American robin', 'scoreboard', 'dome', 'iPod', 'one-armed bandit', 'miniskirt', 'French bulldog', 'carton', 'Tibetan mastiff', 'pencil box', 'king crab, Alaska crab', 'horizontal bar, high bar', 'spider web', 'electric guitar', 'meerkat, mierkat', 'file cabinet', 'consomme', 'jellyfish', 'cuirass', 'black-footed ferret', 'school bus', 'miniature poodle', 'catamaran', 'snorkel', 'oboe', 'worm fence, snake fence', 'African hunting dog', 'golden retriever', 'carousel, carrousel', 'aircraft carrier', 'photocopier', 'Arctic fox, white fox', 'hair slide', 'tile roof', 'Ibizan hound, Ibizan Podenco', 'toucan', 'house finch', 'poncho', 'trifle', 'hourglass', 'fire screen, fireguard', 'white wolf', 'street sign', 'solar dish, solar collector', 'rock beauty', 'komondor', 'bookshop', 'crate', 'theater curtain', 'tank, army tank', 'dugong', 'dalmatian', 'ear, fruit', 'missile', 'bolete', 'orange', 'vase', 'Walker hound', 'lion', 'three-toed sloth', 'lipstick', 'coral reef', 'reel', 'beer bottle', 'green mamba', 'frying pan', 'wok', 'goose', 'rhinoceros beetle', 'yawl', 'clog', 'Saluki Hund', 'chime, bell, gong', 'stage', 'boxer', 'cliff', 'ant', 'cannon', 'harvestman', 'mixing bowl', 'nematode', 'parallel bars', 'garbage truck', 'holster', 'barrel', 'hotdog', 'dishrag'])
stanford_cars_labels = np.array(['AM General Hummer SUV 2000', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012', 'Acura TL Type-S 2008', 'Acura TSX Sedan 2012', 'Acura Integra Type R 2001', 'Acura ZDX Hatchback 2012', 'Aston Martin V8 Vantage Convertible 2012', 'Aston Martin V8 Vantage Coupe 2012', 'Aston Martin Virage Convertible 2012', 'Aston Martin Virage Coupe 2012', 'Audi RS 4 Convertible 2008', 'Audi A5 Coupe 2012', 'Audi TTS Coupe 2012', 'Audi R8 Coupe 2012', 'Audi V8 Sedan 1994', 'Audi 100 Sedan 1994', 'Audi 100 Wagon 1994', 'Audi TT Hatchback 2011', 'Audi S6 Sedan 2011', 'Audi S5 Convertible 2012', 'Audi S5 Coupe 2012', 'Audi S4 Sedan 2012', 'Audi S4 Sedan 2007', 'Audi TT RS Coupe 2012', 'BMW ActiveHybrid 5 Sedan 2012', 'BMW 1 Series Convertible 2012', 'BMW 1 Series Coupe 2012', 'BMW 3 Series Sedan 2012', 'BMW 3 Series Wagon 2012', 'BMW 6 Series Convertible 2007', 'BMW X5 SUV 2007', 'BMW X6 SUV 2012', 'BMW M3 Coupe 2012', 'BMW M5 Sedan 2010', 'BMW M6 Convertible 2010', 'BMW X3 SUV 2012', 'BMW Z4 Convertible 2012', 'Bentley Continental Supersports Conv. Convertible 2012', 'Bentley Arnage Sedan 2009', 'Bentley Mulsanne Sedan 2011', 'Bentley Continental GT Coupe 2012', 'Bentley Continental GT Coupe 2007', 'Bentley Continental Flying Spur Sedan 2007', 'Bugatti Veyron 16.4 Convertible 2009', 'Bugatti Veyron 16.4 Coupe 2009', 'Buick Regal GS 2012', 'Buick Rainier SUV 2007', 'Buick Verano Sedan 2012', 'Buick Enclave SUV 2012', 'Cadillac CTS-V Sedan 2012', 'Cadillac SRX SUV 2012', 'Cadillac Escalade EXT Crew Cab 2007', 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 'Chevrolet Corvette Convertible 2012', 'Chevrolet Corvette ZR1 2012', 'Chevrolet Corvette Ron Fellows Edition Z06 2007', 'Chevrolet Traverse SUV 2012', 'Chevrolet Camaro Convertible 2012', 'Chevrolet HHR SS 2010', 'Chevrolet Impala Sedan 2007', 'Chevrolet Tahoe Hybrid SUV 2012', 'Chevrolet Sonic Sedan 2012', 'Chevrolet Express Cargo Van 2007', 'Chevrolet Avalanche Crew Cab 2012', 'Chevrolet Cobalt SS 2010', 'Chevrolet Malibu Hybrid Sedan 2010', 'Chevrolet TrailBlazer SS 2009', 'Chevrolet Silverado 2500HD Regular Cab 2012', 'Chevrolet Silverado 1500 Classic Extended Cab 2007', 'Chevrolet Express Van 2007', 'Chevrolet Monte Carlo Coupe 2007', 'Chevrolet Malibu Sedan 2007', 'Chevrolet Silverado 1500 Extended Cab 2012', 'Chevrolet Silverado 1500 Regular Cab 2012', 'Chrysler Aspen SUV 2009', 'Chrysler Sebring Convertible 2010', 'Chrysler Town and Country Minivan 2012', 'Chrysler 300 SRT-8 2010', 'Chrysler Crossfire Convertible 2008', 'Chrysler PT Cruiser Convertible 2008', 'Daewoo Nubira Wagon 2002', 'Dodge Caliber Wagon 2012', 'Dodge Caliber Wagon 2007', 'Dodge Caravan Minivan 1997', 'Dodge Ram Pickup 3500 Crew Cab 2010', 'Dodge Ram Pickup 3500 Quad Cab 2009', 'Dodge Sprinter Cargo Van 2009', 'Dodge Journey SUV 2012', 'Dodge Dakota Crew Cab 2010', 'Dodge Dakota Club Cab 2007', 'Dodge Magnum Wagon 2008', 'Dodge Challenger SRT8 2011', 'Dodge Durango SUV 2012', 'Dodge Durango SUV 2007', 'Dodge Charger Sedan 2012', 'Dodge Charger SRT-8 2009', 'Eagle Talon Hatchback 1998', 'FIAT 500 Abarth 2012', 'FIAT 500 Convertible 2012', 'Ferrari FF Coupe 2012', 'Ferrari California Convertible 2012', 'Ferrari 458 Italia Convertible 2012', 'Ferrari 458 Italia Coupe 2012', 'Fisker Karma Sedan 2012', 'Ford F-450 Super Duty Crew Cab 2012', 'Ford Mustang Convertible 2007', 'Ford Freestar Minivan 2007', 'Ford Expedition EL SUV 2009', 'Ford Edge SUV 2012', 'Ford Ranger SuperCab 2011', 'Ford GT Coupe 2006', 'Ford F-150 Regular Cab 2012', 'Ford F-150 Regular Cab 2007', 'Ford Focus Sedan 2007', 'Ford E-Series Wagon Van 2012', 'Ford Fiesta Sedan 2012', 'GMC Terrain SUV 2012', 'GMC Savana Van 2012', 'GMC Yukon Hybrid SUV 2012', 'GMC Acadia SUV 2012', 'GMC Canyon Extended Cab 2012', 'Geo Metro Convertible 1993', 'HUMMER H3T Crew Cab 2010', 'HUMMER H2 SUT Crew Cab 2009', 'Honda Odyssey Minivan 2012', 'Honda Odyssey Minivan 2007', 'Honda Accord Coupe 2012', 'Honda Accord Sedan 2012', 'Hyundai Veloster Hatchback 2012', 'Hyundai Santa Fe SUV 2012', 'Hyundai Tucson SUV 2012', 'Hyundai Veracruz SUV 2012', 'Hyundai Sonata Hybrid Sedan 2012', 'Hyundai Elantra Sedan 2007', 'Hyundai Accent Sedan 2012', 'Hyundai Genesis Sedan 2012', 'Hyundai Sonata Sedan 2012', 'Hyundai Elantra Touring Hatchback 2012', 'Hyundai Azera Sedan 2012', 'Infiniti G Coupe IPL 2012', 'Infiniti QX56 SUV 2011', 'Isuzu Ascender SUV 2008', 'Jaguar XK XKR 2012', 'Jeep Patriot SUV 2012', 'Jeep Wrangler SUV 2012', 'Jeep Liberty SUV 2012', 'Jeep Grand Cherokee SUV 2012', 'Jeep Compass SUV 2012', 'Lamborghini Reventon Coupe 2008', 'Lamborghini Aventador Coupe 2012', 'Lamborghini Gallardo LP 570-4 Superleggera 2012', 'Lamborghini Diablo Coupe 2001', 'Land Rover Range Rover SUV 2012', 'Land Rover LR2 SUV 2012', 'Lincoln Town Car Sedan 2011', 'MINI Cooper Roadster Convertible 2012', 'Maybach Landaulet Convertible 2012', 'Mazda Tribute SUV 2011', 'McLaren MP4-12C Coupe 2012', 'Mercedes-Benz 300-Class Convertible 1993', 'Mercedes-Benz C-Class Sedan 2012', 'Mercedes-Benz SL-Class Coupe 2009', 'Mercedes-Benz E-Class Sedan 2012', 'Mercedes-Benz S-Class Sedan 2012', 'Mercedes-Benz Sprinter Van 2012', 'Mitsubishi Lancer Sedan 2012', 'Nissan Leaf Hatchback 2012', 'Nissan NV Passenger Van 2012', 'Nissan Juke Hatchback 2012', 'Nissan 240SX Coupe 1998', 'Plymouth Neon Coupe 1999', 'Porsche Panamera Sedan 2012', 'Ram C/V Cargo Van Minivan 2012', 'Rolls-Royce Phantom Drophead Coupe Convertible 2012', 'Rolls-Royce Ghost Sedan 2012', 'Rolls-Royce Phantom Sedan 2012', 'Scion xD Hatchback 2012', 'Spyker C8 Convertible 2009', 'Spyker C8 Coupe 2009', 'Suzuki Aerio Sedan 2007', 'Suzuki Kizashi Sedan 2012', 'Suzuki SX4 Hatchback 2012', 'Suzuki SX4 Sedan 2012', 'Tesla Model S Sedan 2012', 'Toyota Sequoia SUV 2012', 'Toyota Camry Sedan 2012', 'Toyota Corolla Sedan 2012', 'Toyota 4Runner SUV 2012', 'Volkswagen Golf Hatchback 2012', 'Volkswagen Golf Hatchback 1991', 'Volkswagen Beetle Hatchback 2012', 'Volvo C30 Hatchback 2012', 'Volvo 240 Sedan 1993', 'Volvo XC90 SUV 2007', 'smart fortwo Convertible 2012'])

class_num_dict={}
class_num_dict['cifar10']=10
class_num_dict['cifar100']=100
class_num_dict['cifar10_full']=10
class_num_dict['cifar100_full']=100
class_num_dict['mini_imagenet'] = len(mini_imagenet_labels)
class_num_dict['stanford_cars'] = len(stanford_cars_labels)

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

def get_large_scale_dataset(name, seed):
    df = pd.read_csv(Path(PATHS[name])/'multimodal_mislabel_split.csv')
    if 'path' in df:
        pass
    else:
        df['path'] = df.apply(lambda x: Path(PATHS[name])/x['filename'], axis = 1)

    train_val_idx, test_idx = train_test_split(df.index, random_state = seed, train_size = 0.75, stratify = df.is_clean)
    train_idx, val_idx = train_test_split(train_val_idx, random_state = seed, train_size = 0.5/0.75, stratify = df.loc[train_val_idx].is_clean)    
    df['split'] = None

    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'

    dfs = {}
    for split in ['train', 'val', 'test']:
        df_split = df.query(f'split == "{split}"')    
        dfs[split] = df_split

    return (LargeScaleDataset(dfs['train'], generic_transform, name),
            LargeScaleDataset(dfs['val'], generic_transform, name),
            LargeScaleDataset(dfs['test'], generic_transform, name))


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
    elif name in ['mscoco', 'flickr30k', 'mimiccxr_caption', 'mmimdb', 'cc3m']:
        assert return_combined_dataset
        cluster_kwargs['clip_model'] = 'biomed_clip' if name == 'mimiccxr_caption' else 'huggingface_clip'
        return get_captioning_dataset(name, data_seed, percent_flips, flip_type, 
                                      data_transform=caption_data_transform,
                                      cluster = cluster_text, 
                                      cluster_kwargs=cluster_kwargs,
                                      )
    elif name in ['stanford_cars', 'mini_imagenet']:
        assert flip_type == 'real'
        return get_large_scale_dataset(name, data_seed)
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