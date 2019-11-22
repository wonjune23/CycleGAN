# CycleGAN
![input](./examples/input.jpg)
![output](./examples/output.jpg)
CycleGAN pytorch custom implementation.

## Prerequisites
- python-anaconda (3.6)

- pytorch

- torchvision : pip install torchvision

- cv2 : pip install opencv-python

- tqdm : pip install tqdm

(optional) if you know how to use wandb..

- wandb : pip install wandb

## Train / Test
to train, run

     python main.py --dataset [dataset_name] --mode train     

to test, run

     python main.py --dataset [dataset_name] --mode test     

[dataset_name] : horse2zebra, maps, apple2orange, cezanne2photo, facades, iphone2dslr_flower, moent2photo, summer2winter_yosemite, ukiyoe2photo, vangogh2photo

The datasets will be automatically downloaded using the original code provided by the authors: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.

to use wandb : --use_wandb 1

to specify the gpu : --gpu [num]
