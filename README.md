
***We are pleased that you have taken an interest in our paper [1] at ECCV 2024, and we hope you will also pay attention to our other two papers [2] and [3] in FSCIL. We plan to integrate the code from these three papers into this project. However, due to the large volume of work, we have initially released the code with the best performance. The remaining parts are still being organized and will be released gradually once they are complete.***

***If you have any questions about the code or the papers, please feel free to contact me.***

[1] Wang X, Ji Z, Liu X, et al. On the Approximation Risk of Few-Shot Class-Incremental Learning[C]//European Conference on Computer Vision. Springer, Cham, 2025: 162-178.

[2] Wang X, Ji Z, Yu Y, et al. Model Attention Expansion for Few-Shot Class-Incremental Learning[J]. IEEE Transactions on Image Processing, 2024, 33: 4419 - 4431.

[3] Wang X, Ji Z, Pang Y, et al. A cognition-driven framework for few-shot class-incremental learning[J]. Neurocomputing, 2024, 600: 128118.


# On the Approximation Risk of Few-Shot Class-Incremental Learning

## Abstract
Few-Shot Class-Incremental Learning (FSCIL) aims to learn new concepts with few training samples while preserving previously acquired knowledge. Although promising performance has been achieved, there remains an underexplored aspect regarding the basic statistical principles underlying FSCIL. Therefore, we thoroughly explore the approximation risk of FSCIL, encompassing both transfer and consistency risks. By tightening the upper bounds of these risks, we derive practical guidelines for designing and training FSCIL models. These guidelines include (1) expanding training datasets for base classes, (2) preventing excessive focus on specific features, (3) optimizing classification margin discrepancy, and (4) ensuring unbiased classification across both base and novel classes. Leveraging these insights, we conduct comprehensive experiments to validate our principles, achieving state-of-the-art performance on three FSCIL benchmark datasets.

## Requirements
conda env create -f environment.yml

## Datasets
We follow [FSCIL](https://github.com/xyutao/fscil) setting to use the same data index_list for training.  
For CIFAR100, the dataset will be download automatically.  
For miniImagenet and CUB200, you can download from [CEC](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN?usp=sharing). Please put the downloaded file under `data/` folder and unzip it:
    
    $ tar -xvf miniimagenet.tar 
    $ tar -xvzf CUB_200_2011.tgz

## Training scripts
mini_imagenet
    $ python train.py -project base -dataset mini_imagenet -arch timm_vit_base_patch16_224 -lr_base 1e-6 -epochs_base 30 -warmup_rate 0.6667 > output.txt


cub200
    $ python train.py -project base -dataset cub200  -arch timm_vit_base_patch16_224 -lr_base 1e-5 -epochs_base 30 -warmup_rate 0.6667 > output.txt


cifar100
    $ python train.py -project base -dataset cifar100  -arch timm_vit_base_patch16_224 -lr_base 1e-5 -epochs_base 10 -warmup_rate 0.5 > output.txt


## Acknowledgement

The code is built on the [CEC] (https://github.com/icoz69/CEC-CVPR2021)
