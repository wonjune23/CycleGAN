# CycleGAN
![input](./examples/input.jpg)
![output](./examples/output.jpg)

CycleGAN pytorch custom implementation.

## Prerequisites

- Linux

- python-anaconda (3.6)

- pytorch

- torchvision : pip install torchvision

- cv2 : pip install opencv-python

- tqdm : pip install tqdm

(optional) if you know how to use wandb, you can log your training.

- wandb : pip install wandb

## Train / Test

- Clone this repo and change directory:


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

### Possible Error
- If you have ROS installed in your system and it causes error importing cv2, you need to add these lines before importing cv2 in DataLoader.py.


      import sys
      sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
