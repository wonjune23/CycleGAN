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
from models import *
from DataLoader import CycleGANDataset
import cv2
from PIL import Image

import wandb

batch_size = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
target_size = 128
e_identity = 0
e_cycle = 10

wandb.init(project = "cycleGAN")

def train():


    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)
    #G = torch.nn.DataParallel(G)
    #D = torch.nn.DataParallel(D)
    #net.load_state_dict(torch.load('./parameters/paramters'))
    #print('model restored!')

    G_A2B_optim = torch.optim.Adam(G_A2B.parameters(), lr = 0.0001)
    G_B2A_optim = torch.optim.Adam(G_B2A.parameters(), lr=0.0001)
    D_A_optim = torch.optim.Adam(D_A.parameters(), lr = 0.0001)
    D_B_optim = torch.optim.Adam(D_B.parameters(), lr=0.0001)

    train_step = 0

    L2Loss = nn.MSELoss()
    L1Loss = nn.L1Loss()

    for epoch in range(200):
        for i, (imgA, imgB) in enumerate(trainloader):

            for Disc in range(1):
                train_step += 1
                # Discriminator steps
                D_A.zero_grad()
                D_B.zero_grad()

                realA = imgA.to(device)
                realB = imgB.to(device)

                fakeB = G_A2B(realA)
                fakeA = G_B2A(realB)

                D_A_real = D_A(realA)
                D_A_fake = D_A(fakeA)
                D_B_real = D_B(realB)
                D_B_fake = D_B(fakeB)

                y_real = torch.ones_like(D_A_real)
                y_fake = -torch.ones_like(D_A_real)

                D_A_real_loss = L2Loss(D_A_real, y_real)
                D_B_real_loss = L2Loss(D_B_real, y_real)
                D_A_fake_loss = L2Loss(D_A_fake, y_fake)
                D_B_fake_loss = L2Loss(D_B_fake, y_fake)

                D_A_GANLoss = (D_A_real_loss + D_A_fake_loss)/2
                D_B_GANLoss = (D_B_real_loss + D_B_fake_loss)/2

                D_A_GANLoss.backward()
                D_B_GANLoss.backward()

                D_A_optim.step()
                D_B_optim.step()

                if train_step % 1 == 0:

                    G_A2B.zero_grad()
                    G_B2A.zero_grad()

                    fakeB = G_A2B(realA)
                    fakeA = G_B2A(realB)

                    D_A_fake = D_A(fakeA)
                    D_B_fake = D_B(fakeB)

                    G_A2B_GANLoss = L2Loss(D_B_fake, y_real)
                    G_B2A_GANLoss = L2Loss(D_A_fake, y_real)

                    B_idt = G_A2B(realB)
                    A_idt = G_B2A(realA)

                    G_A2B_iLoss = L1Loss(B_idt, realB)
                    G_B2A_iLoss = L1Loss(A_idt, realA)

                    A_Cycle = G_B2A( G_A2B(realA))
                    B_Cycle = G_A2B( G_B2A(realB))
                    A_CycleLoss = L1Loss( A_Cycle , realA)
                    B_CycleLoss = L1Loss( B_Cycle , realB)

                    G_loss = (G_A2B_GANLoss + G_B2A_GANLoss) + e_identity*(G_A2B_iLoss + G_B2A_iLoss) + e_cycle*(A_CycleLoss+B_CycleLoss)
                    G_loss.backward()
                    G_A2B_optim.step()
                    G_B2A_optim.step()

                    if train_step % 5 == 0:
                        wandb.log({"fakeB": [wandb.Image((255*np.array(fakeB[0].transpose(0,1).transpose(1,2).cpu().detach() * 2) + 0.5))],
                                   "realA": [wandb.Image((255 * np.array(
                                       realA[0].transpose(0, 1).transpose(1, 2).cpu().detach() * 2) + 0.5))],
                                   "realB": [wandb.Image((255 * np.array(
                                       realB[0].transpose(0, 1).transpose(1, 2).cpu().detach() * 2) + 0.5))],
                                   "B_cycle": [wandb.Image((255 * np.array(
                                       B_idt[0].transpose(0, 1).transpose(1, 2).cpu().detach() * 2) + 0.5))],
                                   "G_A2B_GANLoss": G_A2B_GANLoss.detach().cpu().numpy(),
                                   "D_B_GANLoss": D_B_GANLoss.detach().cpu().numpy(),
                                   "B_CycleLoss": B_CycleLoss.detach().cpu().numpy()
                                   })

        print('epoch{}'.format(epoch + 1))


    print('####### ### #     # ###  #####  #     # ####### ######     ### ### \n#        #  ##    #  #  #     # #     # #       #     #    ### ###\n#        #  # #   #  #  #       #     # #       #     #    ### ###\n#####    #  #  #  #  #   #####  ####### #####   #     #     #   #\n#        #  #   # #  #        # #     # #       #     #\n#        #  #    ##  #  #     # #     # #       #     #    ### ###\n#       ### #     # ###  #####  #     # ####### ######     ### ###')

if __name__  == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CycleGANDataset(root='./datasets/horse2zebra', transform=transform, train_flag='train')
    # estset = torchvision.datasets.MNIST(root = './MNIST/test', train = False, download = True, transform = transform)

    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    # testloader = DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)
    train()