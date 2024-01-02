# [TIP 2023] 4D LUT
This is the official PyTorch implementation of the paper [4D LUT: Learnable Context-Aware 4D Lookup Table for Image Enhancement](https://arxiv.org/abs/2209.01749).

## Overview
<img src="./4D LUT_overview.jpg" width=100%>

## Contribution
* We propose a novel learnable context-aware 4-dimensional lookup table (4D~LUT), which first extends the lookup table architecture into a 4-dimensional space and achieve content-dependent image enhancement without a significant increase in computational costs.
* The extensive experiments demonstrate that the proposed 4D~LUT can obtain more accurate results and significantly outperform existing SOTA methods in three widely-used image enhancement benchmarks.

## Requirements and dependencies
* CUDA 11.4
* GCC 7.5.0
* python 3.6 (recommend to use [Anaconda](https://www.anaconda.com/))
* pytorch == 1.9.0
* torchvision == 0.10.0

## Train
1. Clone this github repo
```
git clone https://github.com/ChengxuLiu/4DLUT.git
cd 4DLUT
```
2. Build:
```
cd quadrilinear_cpp
sh setup.sh
```
3. Prepare training dataset and modify "datasetPath" in `./dataset.py`
4. Run training
```
python train.py
```
5. The models are saved in `./saved_models`


## Citation
If you find the code and pre-trained models useful for your research, please consider citing our paper. :blush:
```
@article{liu20234d,
  title={4D LUT: learnable context-aware 4d lookup table for image enhancement},
  author={Liu, Chengxu and Yang, Huan and Fu, Jianlong and Qian, Xueming},
  journal={IEEE Transactions on Image Processing},
  volume={32},
  pages={4742--4756},
  year={2023},
  publisher={IEEE}
}
``` 

## Contact
If you meet any problems, please describe them in issues or contact:
* Chengxu Liu: <liuchx97@gmail.com>


