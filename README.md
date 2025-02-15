# CycleGAN
<img src='examples/input.jpg' width=200> <img src='examples/output.jpg' width=200>

CycleGAN pytorch custom implementation.

## Prerequisites

- Linux

- python-anaconda (3.6)

- pytorch

- torchvision : pip install torchvision

- cv2 : pip install opencv-python

- tqdm : pip install tqdm

(optional) 

- wandb : pip install wandb

## Train / Test

- Clone this repo:


      git clone https://github.com/wonjune23/CycleGAN.git     
      cd CycleGAN
- to train, run


      python main.py --dataset [dataset_name] --mode train     

- to test, run


      python main.py --dataset [dataset_name] --mode test     


[dataset_name] : horse2zebra, maps, apple2orange, cezanne2photo, facades, iphone2dslr_flower, moent2photo, summer2winter_yosemite, ukiyoe2photo, vangogh2photo

The datasets will be automatically downloaded using the original code provided by the authors: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.

to use wandb : --use_wandb 1

to specify the gpu : --gpu [num]

You can see the example training processes here: https://app.wandb.ai/wonjune/cycleGAN?workspace=user-wonjune
