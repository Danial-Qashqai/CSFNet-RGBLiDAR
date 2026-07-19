## Overview

This repository presents **CSFNet-RGBLiDAR**, a real-time RGB-LiDAR semantic segmentation model trained with projected LiDAR data on a subset of the KITTI-360 dataset.

This work extends our previous **CSFNet** framework to the RGB-LiDAR domain. The original CSFNet implementation is available at [CSFNet](https://github.com/Danial-Qashqai/CSFNet).

## Results
We provide the pre-trained weights of our RGB-LiDAR semantic segmentation model.

### Validation Results on a Subset of the KITTI-360 Dataset (19 Categories)
| Architecture | Backbone | Params(M) | FPS | mIoU.half | Weight |
|:---:|:---:|:---:|:---:|:---:|:---:|
| CSFNet-1 | STDC1 | 11.31 | 106.2 | 47.3 | [Kaggle](https://drive.google.com/file/d/1yK1Fg7NX1zryVDQTbzIDVGnn8prxLsjY/view?usp=sharing)

![sample.png]([https://github.com/Danial-Qashqai/CSFNet/blob/main/figures/Figure_4.png](https://github.com/Danial-Qashqai/CSFNet-RGBLiDAR/blob/main/figures/sample.png))
*Visual results of CSFNet-1 on KITTI-360. From top to bottom: RGB input, LiDAR projection, and prediction output..*


