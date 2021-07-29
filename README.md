# Multi-View Attention (MVA) Monocular Depth Estimation
Implementation of MVA monocular depth estimation.   
Reference code : [DenseDepth](https://github.com/ialhashim/DenseDepth)

## Requirements
- Tensorflow 2.2
- Numpy
- Pilow
- Matplotlib
- tqdm

## Pre-trained model
* [Trained by NYU RGB-D V2](https://drive.google.com/uc?export=download&id=1k8McRE2vOtrkHmG9ZU6Cd-IUDtr2Fbbv) (650 MB)

## Usage
- #### Use pre-trained model   
    1. Download pre-trained model weights from above download link above.
    2. Save downloaded model weights to `checkpoint` directory.

- #### Use jupyter notebook (example.ipynb)
    1. Locate model weights at `checkpoin` directory.
    2. Locate the own test image at `examples` directory.
    3. Go to `example.ipynb`

- #### Training
    - Prepare the dataset for training (we used the dataset created by DenseDepth, See [DenseDepth](https://github.com/ialhashim/DenseDepth))
    - Run following command.   
    ```python train.py --bs 4 --lr 0.0001 --epochs 20```
    
## Results

## To-Do List
1. Add ```test.py```
2. Add Results
3. Update my code