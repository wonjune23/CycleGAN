import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.modules.conv import _ConvNd

import os
'''
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
'''
import cv2
from PIL import Image

batch_size = 1

target_size = 128

class CycleGANDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, train_flag='train'):
        assert ((train_flag == 'train') | (train_flag == 'test')) , 'train_flag is wrong'
        self.train_flag = train_flag
        self.root = os.path.expanduser(root+'/'+train_flag)
        self.transform = transform
        self.imagesNameA = os.listdir(self.root+'A')
        self.imagesNameB = os.listdir(self.root + 'B')


    def __len__(self):
        return min(len(self.imagesNameA), len(self.imagesNameB))

    def __getitem__(self, idx):
        print((self.imagesNameA[idx]))
        print((self.imagesNameB[idx]))
        imgA = cv2.imread(self.root + 'A/' + self.imagesNameA[idx])
        imgA = cv2.resize(imgA, dsize=(target_size,target_size), interpolation=cv2.INTER_CUBIC)
        imgA = np.array(imgA)
        imgA = self.transform(imgA)
        imgA = imgA[[2,1,0],:,:]

        imgB = cv2.imread(self.root + 'B/' + self.imagesNameB[idx])
        imgB = cv2.resize(imgB, dsize=(target_size, target_size), interpolation=cv2.INTER_CUBIC)
        imgB = np.array(imgB)
        imgB = self.transform(imgB)
        imgB = imgB[[2, 1, 0], :, :]

        return imgA, imgB


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CycleGANDataset(root = './datasets/horse2zebra', transform = transform, train_flag = 'train')
#estset = torchvision.datasets.MNIST(root = './MNIST/test', train = False, download = True, transform = transform)

trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 1)
#testloader = DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)


if __name__  == '__main__':

    train_step = 1
    for epoch in range(1):
        for i, (imgA, imgB) in enumerate(trainloader):
            print(imgA)
            print(imgB)


