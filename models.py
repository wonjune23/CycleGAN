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

class ResBlock(nn.Module):
    def __init__(self, inC, kernel_size, stride, pad_size):
        super(ResBlock, self).__init__()

        self.conv1 = ConvBlock(inC, inC, kernel_size, stride, pad_size)
        self.conv2 = ConvBlock(inC, inC, kernel_size, stride, pad_size)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        return input + x

class ConvBlock(nn.Module):
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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.first_layer = nn.Sequential(ConvBlock(3, 32, 3, 1, 1),
                                          ResBlock(32, 3, 1, 1),
                                          ResBlock(32, 3, 1, 1))

        inC = 32

        downsample = []
        for i in range(3):
            downsample += [nn.AvgPool2d(2)]
            downsample += [ConvBlock(inC, inC*2, 3, 1, 1)]
            downsample += [ResBlock(inC*2, 3, 1, 1)]
            inC *= 2

        self.Down = nn.Sequential( * downsample )

        upsample = []
        for i in range(3):
            upsample += [nn.Upsample(scale_factor = 2)]
            upsample += [ConvBlock(inC, inC//2, 3, 1, 1)]
            upsample += [ResBlock(inC//2, 3, 1, 1)]
            inC = inC//2

        self.Up = nn.Sequential( * upsample )

        self.last_layers = nn.Sequential( ResBlock(inC, 3, 1, 1),
                                          ResBlock(inC, 3, 1, 1),
                                          ConvBlock(inC, 3, 3, 1, 1))

    def forward(self, input):
        # input : B, 3, target_size, target_size
        first = self.first_layer(input)
        down = self.Down(first)
        up = self.Up(down) + first # skip connection

        final = self.last_layers(up)

        return F.tanh(final) # returns -1 ~ 1 output and latent vector

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        # patch discriminator..
        self.first_layer = ConvBlock(3, 32, 3, 1, 1)
        inC = 32

        downsample = []
        for i in range(4):
            downsample += [nn.AvgPool2d(2)]
            downsample += [ConvBlock(inC, inC*2, 1, 1, 1)]
            downsample += [ResBlock(inC*2, 1, 1, 0)]
            inC *= 2

        self.Down = nn.Sequential( * downsample )

        self.last_layers = nn.Sequential( ResBlock(inC, 1, 1, 0),
                                          ConvBlock(inC, 1, 1, 1, 1))

    def forward(self, input):
        #x = input.reshape(-1, 3, target_size, target_size)
        x = self.first_layer(input)
        x = self.Down(x)
        x = self.last_layers(x)

        return x # no sigmoid for WGAN

if __name__  == '__main__':
    G = Generator().cuda()
    D = Discriminator().cuda()
    #G = torch.nn.DataParallel(G)
    #D = torch.nn.DataParallel(D)
    #net.load_state_dict(torch.load('./parameters/paramters'))
    #print('model restored!')
    Goptimizer = torch.optim.Adam(G.parameters(), lr = 0.00003)
    Doptimizer = torch.optim.Adam(D.parameters(), lr = 0.00003)

    train_step = 0
    mean_print = 1

    D_train_loss = torch.autograd.Variable(torch.Tensor([1.0]))
    G_train_loss = torch.autograd.Variable(torch.Tensor([3.0]))

    Loss = nn.BCELoss()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CycleGANDataset(root='./datasets/horse2zebra', transform=transform, train_flag='train')
    # estset = torchvision.datasets.MNIST(root = './MNIST/test', train = False, download = True, transform = transform)

    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    # testloader = DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)


    if __name__ == '__main__':

        train_step = 1
        for epoch in range(1):
            for i, (imgA, imgB) in enumerate(trainloader):
                print(G(imgA.cuda()))
                print(D(G(imgA.cuda())))
