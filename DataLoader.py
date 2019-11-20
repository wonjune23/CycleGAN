import torch
from torch.utils.data import DataLoader
import numpy as np
import os
''''''
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
''''''
import cv2

class CycleGANDataset(torch.utils.data.Dataset):
    def __init__(self, args, transform):
        assert ((args.mode == 'train') | (args.mode == 'test')) , 'train_flag is wrong'
        self.train_flag = args.mode
        root = f'./datasets/{args.dataset}'
        if not os.path.isdir(root):
            print(f"\ndataset {args.dataset} doesn't exist: start downloading the dataset\n")
            os.system(f"bash ./datasets/download_cyclegan_dataset.sh {args.dataset}")
            print(f"\ndataset {args.dataset} download complete : start network {args.mode}ing now.\n")
        self.root = os.path.expanduser(root+'/'+args.mode)
        self.transform = transform
        self.imagesNameA = os.listdir(self.root + 'A')
        self.imagesNameB = os.listdir(self.root + 'B')
        self.target_size = args.target_size

    def __len__(self):
        return min(len(self.imagesNameA), len(self.imagesNameB))

    def __getitem__(self, idx):
        imgA = cv2.imread(self.root + 'A/' + self.imagesNameA[idx])
        imgA = cv2.resize(imgA, dsize=(self.target_size,self.target_size), interpolation=cv2.INTER_CUBIC)
        imgA = np.array(imgA)
        imgA = self.transform(imgA)
        imgA = imgA[[2,1,0],:,:]

        imgB = cv2.imread(self.root + 'B/' + self.imagesNameB[idx])
        imgB = cv2.resize(imgB, dsize=(self.target_size, self.target_size), interpolation=cv2.INTER_CUBIC)
        imgB = np.array(imgB)
        imgB = self.transform(imgB)
        imgB = imgB[[2, 1, 0], :, :]

        return imgA, imgB