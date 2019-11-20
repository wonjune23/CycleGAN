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
        inC = 64
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(3, inC, 7, 1, 0),
                                         nn.InstanceNorm2d(inC),
                                         nn.ReLU(True))

        downsample = []
        for i in range(2):
            downsample += [nn.Conv2d(inC, inC*2, 3, 2, 1)]
            downsample += [nn.InstanceNorm2d(inC*2)]
            downsample += [nn.ReLU(True)]
            inC *= 2
        self.Down = nn.Sequential( * downsample )

        resblocks = []
        for i in range(6):
            resblocks += [ResBlock(inC,3, 1, 1)]
        self.Residual = nn.Sequential(*resblocks)

        upsample = []
        for i in range(2):
            upsample += [nn.ConvTranspose2d(inC, inC//2, 3, 2, 1, output_padding=1)]
            upsample += [nn.InstanceNorm2d(inC//2)]
            upsample += [nn.ReLU(True)]
            inC = inC//2

        self.Up = nn.Sequential( * upsample )

        self.last_layers = nn.Sequential( nn.ReflectionPad2d(3),
                                          nn.Conv2d(inC, 3, 7, 1, 0))

    def forward(self, input):
        # input : B, 3, target_size, target_size
        first = self.first_layer(input)
        down = self.Down(first)
        res = self.Residual(down)
        up = self.Up(res)
        final = self.last_layers(up)

        return F.tanh(final) # returns -1 ~ 1 output and latent vector

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        # patch discriminator..
        inC = 64
        self.first_layer = nn.Sequential( nn.Conv2d(3, inC, 4, 2, 1),
                                          nn.LeakyReLU(0.2, True))

        layers = []
        for i in range(3):
            layers += [nn.Conv2d(inC, min(inC//2,8), 4, 2, 1)]
            layers += [nn.InstanceNorm2d(min(inC//2,8))]
            layers += [nn.LeakyReLU(0.2,True)]
            inC = min(inC//2,8)

        self.Seq = nn.Sequential( * layers )

        self.last_layers = nn.Sequential( nn.Conv2d(inC, min(inC//2,8),4, 1, 1),
                                          nn.InstanceNorm2d(min(inC//2,8)),
                                          nn.LeakyReLU(0.2,True),
                                          nn.Conv2d(min(inC//2,8), 1, 4, 1, 1))

    def forward(self, input):
        #x = input.reshape(-1, 3, target_size, target_size)
        x = self.first_layer(input)
        x = self.Seq(x)
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
