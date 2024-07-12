
### MSCOCO

1. Download and extract the COCO 2014 Train images and 2014 Val images from [here](https://cocodataset.org/#download).

2. Download the Karpathy split for COCO from [here](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits).

3. Run `notebooks/preprocess_mscoco.ipynb`, updating paths at the top of the notebook.

4. Update the `PATHS` variable at the top of `libs/datasets/utils.py`.


### Flickr30k

1. Download the Flickr30k images from [here](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).

2. Download the Karpathy split for Flickr30k from [here](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits).

3. Run `notebooks/preprocess_flickr30k.ipynb`, updating paths at the top of the notebook.

4. Update the `PATHS` variable at the top of `libs/datasets/utils.py`.


### MMIMDB

1. Download the MMIMDB dataset from [here](https://www.kaggle.com/datasets/eduardschipatecua/mmimdb).

2. Run `notebooks/preprocess_mmimdb.ipynb`, updating paths at the top of the notebook.

3. Update the `PATHS` variable at the top of `libs/datasets/utils.py`.


### MIMIC-CXR

1. [Obtain access](https://mimic-cxr.mit.edu/about/access/) to the MIMIC-CXR-JPG Database Database on PhysioNet and download the [dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). 

2. Download and unzip the `mimic-cxr-reports.zip` file from [this repository](https://physionet.org/content/mimic-cxr/2.0.0/).

3. Run `notebooks/preprocess_mimiccxr.ipynb`, updating paths at the top of the notebook.

4. Update the `PATHS` variable at the top of `libs/datasets/utils.py`.
