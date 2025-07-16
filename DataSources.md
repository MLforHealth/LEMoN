
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



### StanfordCars + Mini-ImageNet

1. Download the Image URLs with annotations [here](https://google.github.io/controlled-noisy-web-labels/download.html).

2. Download the images, e.g. with the [img2dataset library](https://github.com/rom1504/img2dataset).

    1. Write a temporary csv file that the library can load:
        ```
        import pandas as pd
        from pathlib import Path

        df = pd.read_json('/PATH/ImageNetRed/dataset_no_images/mini-imagenet-annotations.json')
        df = pd.DataFrame(df['data'].apply(lambda x: x[0]).tolist())

        out_dir = Path('/PATH/ImageNetRed/dataset_no_images/mini-imagenet/images')
        out_dir.mkdir(exist_ok = True)

        df = df.rename(columns = {
            'image/uri': 'url'
        })

        df.to_csv('./temp_mini_imagenet.csv')
        ```


    2. Run the following:

        ```
        img2dataset --input_format csv --url_list=/PATH/temp_mini_imagenet.csv --output_folder=/PATH/ImageNetRed/dataset_no_images/mini-imagenet/images --thread_count=64 --image_size=256 --resize_mode keep_ratio
        ```

3. Run `notebooks/preprocess_imagenet_red.ipynb`, updating paths at the top of the notebook.

4. Update the `PATHS` variable at the top of `libs/datasets/utils.py`.

