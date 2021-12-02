# Multi-View Attention (MVA) Monocular Depth Estimation
Tensorflow 2 Implementation of MVA monocular depth estimation.   
Reference code : [DenseDepth](https://github.com/ialhashim/DenseDepth)

## Requirements
- Tensorflow 2.2
- Numpy
- Pilow
- Matplotlib
- Scikit-image 0.15.0
- tqdm

## Pre-trained model
* [Trained by NYU RGB-D V2](https://drive.google.com/uc?export=download&id=1k8McRE2vOtrkHmG9ZU6Cd-IUDtr2Fbbv) (650 MB)

## Dataset
- We use NYU Detph V2 dataset and refer to DenseDepth, See [DenseDepth](https://github.com/ialhashim/DenseDepth) Data section

## Usage
- #### Use pre-trained model   
    1. Download pre-trained model weights from above download link above.
    2. Save downloaded model weights to `checkpoints` directory or `your_own_path`.

- #### Use jupyter notebook (example.ipynb)
    1. Make direcotry `checkpoints` directory.
    2. Locate model weights at `checkpoints` directory.
    3. Locate the own test image at `examples` directory.
    4. Go to `example.ipynb`

## Train & Test network
- #### Train
    - Prepare the dataset for training. (we used the NYU V2 dataset)
    - Run following command.   
    ```python train.py --bs 4 --lr 0.0001 --epochs 20```

- #### Test
    - Prepare your test images in `examples` directory or your own directory.
    - Make `checkpoints` directory.
    - Locate model weights at `checkpoints` directory.
    - Run following command.   
    ```
    python test.py \
        --model_weights /your/own/path \
        --images_dir /your/own/path \
        --results_dir /your/own/path \
        --gpu your_gpu_number
    ```
    
## Results
<p align="center"><img src="https://user-images.githubusercontent.com/55485826/127944218-2c72c094-2bc6-4b15-8241-f7e36e25dbde.png"></p>