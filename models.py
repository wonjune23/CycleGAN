import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.modules.conv import _ConvNd
from DataLoader import CycleGANDataset
import os
from PIL import Image

class ResBlock(nn.Module): # Residual Block. it consists of 2 ConvBlocks and residual connection.
    def __init__(self, inC, kernel_size, stride, pad_size):
        super(ResBlock, self).__init__()

        self.conv1 = ConvBlock(inC, inC, kernel_size, stride, pad_size)
        self.conv2 = ConvBlock(inC, inC, kernel_size, stride, pad_size)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        return input + x

class ConvBlock(nn.Module): # ConvBLock. it consists of reflection padding, convolution, instance normalization, and ReLU.
    def __init__(self, inC, outC, kernel_size, stride, pad_size):
        super(ConvBlock, self).__init__()

        self.pad1 = nn.ReflectionPad2d(pad_size)
        self.conv1 = nn.Conv2d(inC, outC, kernel_size, stride)
        self.norm1 = nn.InstanceNorm2d(outC)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.pad1(input)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x

class Generator(nn.Module): # Generator network. it consists of several downsampling layers, 6 residual blocks, and upsampling layers.
    def __init__(self):
        super(Generator, self).__init__()
        inC = 64 # number of channels in first conv layer
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), # first layer.
                                         nn.Conv2d(3, inC, 7, 1, 0),
                                         nn.InstanceNorm2d(inC),
                                         nn.ReLU(True))

        downsample = [] # downsampling layers.
        for i in range(2):
            downsample += [nn.Conv2d(inC, inC*2, 3, 2, 1)]
            downsample += [nn.InstanceNorm2d(inC*2)]
            downsample += [nn.ReLU(True)]
            inC *= 2
        self.Down = nn.Sequential( * downsample )

        resblocks = [] # residual block layers.
        for i in range(6):
            resblocks += [ResBlock(inC,3, 1, 1)]
        self.Residual = nn.Sequential(*resblocks)

        upsample = [] # upsampling layers.
        for i in range(2):
            upsample += [nn.ConvTranspose2d(inC, inC//2, 3, 2, 1, output_padding=1)]
            upsample += [nn.InstanceNorm2d(inC//2)]
            upsample += [nn.ReLU(True)]
            inC = inC//2

        self.Up = nn.Sequential( * upsample )

        self.last_layers = nn.Sequential( nn.ReflectionPad2d(3), # The last layer to map the feature maps into image space.
                                          nn.Conv2d(inC, 3, 7, 1, 0))

    def forward(self, input):
        # input : B, 3, target_size, target_size
        first = self.first_layer(input)
        down = self.Down(first)
        res = self.Residual(down)
        up = self.Up(res)
        final = self.last_layers(up)

        return torch.tanh(final) # return image in [-1 , 1] (the images are normalized to lie in [-1,1] in the prepcoess step)

class Discriminator(nn.Module): # Discriminator step. it is 70x70 patch discriminator.

    def __init__(self):
        super(Discriminator, self).__init__()

        inC = 64 # number of channels in first conv layer
        self.first_layer = nn.Sequential( nn.Conv2d(3, inC, 4, 2, 1), # the first layer.
                                          nn.LeakyReLU(0.2, True))

        layers = [] # mid layers.
        for i in range(3):
            layers += [nn.Conv2d(inC, min(inC//2,8), 4, 2, 1)]
            layers += [nn.InstanceNorm2d(min(inC//2,8))]
            layers += [nn.LeakyReLU(0.2,True)]
            inC = min(inC//2,8)

        self.Seq = nn.Sequential( * layers )

        self.last_layers = nn.Sequential( nn.Conv2d(inC, min(inC//2,8),4, 1, 1), # last conv block.
                                          nn.InstanceNorm2d(min(inC//2,8)),
                                          nn.LeakyReLU(0.2,True),
                                          nn.Conv2d(min(inC//2,8), 1, 4, 1, 1)) # At the very last, map the feature maps into one-channel score map.

    def forward(self, input):
        #x = input.reshape(-1, 3, target_size, target_size)
        x = self.first_layer(input)
        x = self.Seq(x)
        x = self.last_layers(x)

        return x # no sigmoid for LSGAN.