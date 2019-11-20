# CycleGAN
CycleGAN pytorch custom implementation.

## Requirements
- python(3.6)
- Anaconda python packages
- pytorch
- torchvision : pip install torchvision
- cv2 : pip install opencv-python
(optional) if you know how to use wandb..
- wandb : pip install wandb

## Train
python main.py --dataset [dataset_name] --mode [mode]
[dataset_name] examples include horse2zebra, maps, etc.
[mode] only takes train or test.
