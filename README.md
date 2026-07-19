## Overview

This repository presents **CSFNet-RGBLiDAR**, a real-time RGB-LiDAR semantic segmentation model trained with projected LiDAR data on a subset of the KITTI-360 dataset.

This work extends our previous **CSFNet** framework to the RGB-LiDAR domain. The original CSFNet implementation is available at [CSFNet](https://github.com/Danial-Qashqai/CSFNet).

## Results
We provide the pre-trained weights of our RGB-LiDAR semantic segmentation model.

### Validation Results on a Subset of the KITTI-360 Dataset (19 Categories)
| Architecture | Backbone | Params(M) | FPS | mIoU.half | Weight |
|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | 11.31 | 106.2 | 47.3 | [Kaggle](https://www.kaggle.com/datasets/danialqashqai/best-csfnet-1-weight-rgbl)

![sample.png](https://github.com/Danial-Qashqai/CSFNet-RGBLiDAR/blob/main/figures/sample.png)
*Visual results of CSFNet-1 on KITTI-360. From top to bottom: RGB input, LiDAR projection, prediction output, and ground truth.*

## Note
Due to hardware limitations, we trained the model on a subset of the KITTI-360 dataset. Training on the full dataset can achieve higher accuracy.

## Usage
### Installation
Clone repository:
Please navigate to the cloned directory.
```
git clone https://github.com/Danial-Qashqai/CSFNet-RGBLiDAR
cd /path/to/this/repository
```
Requirements:
we are using Python 3.12.13, Torch 2.10.0, torchvision 0.25.0 and CUDA 12.8.

Install pytorch, cuda and cudnn, then install other dependencies via:
```shell
pip install -r requirements.txt
```
### Datasets
For convenience, we provide the KITTI-360 subset dataset at the following links:
- RGB images: [Kaggle](https://www.kaggle.com/datasets/danialqashqai/kitti360-mini-rgb)
- Projected LiDAR data: [Kaggle](https://www.kaggle.com/datasets/danialqashqai/kitti360-mini-projected-lidar)
- Semantic labels: [Kaggle](https://www.kaggle.com/datasets/danialqashqai/kitti360-mini-semanticlabel)

### Pre-trained ImageNet Backbones
The pre-trained weight for the [STDC1](https://github.com/MichaelFan01/STDC-Seg) backbone is available at the following links:
- [STDC1](https://drive.google.com/file/d/1xR7Hg0CQcGyCFRgoF6vuhFNClE4ACpF_/view?usp=sharing)

  
## Training

* To train **CSFNet-1**, run:
```
python train.py \
    --batch_size 16 \
    --batch_size_valid 8 \
    --num_workers 2  \
    --val_seq "seq_0007"  \
    --num_gpus 2 \
    --network "CSFNet-1" \
    --num_classes 19 \
    --epochs 300 \
    --lr 0.02  \
    --pretrained True \
    --backbone_path "./STDCNet813M_73.91.tar"  \
    --crop_H 376 \
    --crop_W 1408 \
    --img_path "./kitti360-mini-rgb/RGB"  \
    --lidar_path "./kitti360-mini-projected-lidar/Lidar"  \
    --label_path "./kitti360-mini-semanticlabel/Label"  
```

## Evaluation
* To Evaluate **CSFNet-1**, run:
```
python eval.py \
    --batch_size_valid 8 \
    --num_workers 2  \
    --val_seq "seq_0007"  \
    --num_gpus 1 \
    --network "CSFNet-1" \
    --num_classes 19 \
    --weight_path "./best_CSFNet-1_weight.pth" \
    --img_path "./kitti360-mini-rgb/RGB"  \
    --lidar_path "./kitti360-mini-projected-lidar/Lidar"  \
    --label_path "./kitti360-mini-semanticlabel/Label"  
```

## Inference
* To perform inference with **CSFNet-1**, run:
```
python inference.py \
    --weight_path "./best_CSFNet-1_weight.pth" \
    --single_img_path "./Samples/0000001615.png)"  \
    --single_lidar_path "./Samples/0000001615.npy"  
```


## Contact

Danial Qashqai：danialqashqai99@gmail.com




