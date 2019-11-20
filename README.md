# CycleGAN
CycleGAN pytorch custom implementation.

## Prerequisites
- python-anaconda (3.6)

- pytorch

- torchvision : pip install torchvision

- cv2 : pip install opencv-python

- tqdm : pip install tqdm

(optional) if you know how to use wandb..

- wandb : pip install wandb

## Train
    python main.py --dataset [dataset_name] --mode [mode]     

[dataset_name] : horse2zebra, maps, etc.

[mode] : train / test.

to use wandb : --use_wandb 1

to specify the gpu : --gpu [num]
