import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.modules.conv import _ConvNd
import visdom
import os
import cv2
from PIL import Image

batch_size = 4

target_size = 128

vis = visdom.Visdom( port='8097', env = 'DCGAN_3')

def train():

if __name__  == '__main__':
    G = Generator()
    D = Descriminator()
    G = torch.nn.DataParallel(G)
    D = torch.nn.DataParallel(D)
    #net.load_state_dict(torch.load('./parameters/paramters'))
    #print('model restored!')
    Goptimizer = torch.optim.RMSprop(G.parameters(), lr = 0.00003)
    Doptimizer = torch.optim.RMSprop(D.parameters(), lr = 0.00003)

    train_step = 0
    mean_print = 1

    D_train_loss = torch.autograd.Variable(torch.Tensor([1.0]))
    G_train_loss = torch.autograd.Variable(torch.Tensor([3.0]))

    Loss = nn.BCELoss()

    for epoch in range(2000000):
        for i, data in enumerate(trainloader):

            for critic in range(1):
                train_step += 1

                D.zero_grad()
                input, labels = data
                input, labels = Variable(input.cuda()), Variable(labels.cuda())
                current_batch_size = len(input[:,0,0,0])
                y_real = Variable(torch.ones(current_batch_size).cuda())
                y_fake = Variable(torch.zeros(current_batch_size).cuda())

                D_result = D(input)

                D_real_loss = Loss(D_result, y_real)

                D_real_score = D_result

                v1 = Variable(torch.rand(current_batch_size, 100) * 2 - 1)
                G_result1 = G(v1)
                D_result = D(G_result1)
                D_fake_loss = Loss(D_result, y_fake)

                D_fake_score = D_result

                #D_train_loss = -(D_real_loss - D_fake_loss), when D_real_loss = - Loss()

                D_train_loss = D_real_loss + D_fake_loss

                D_train_loss.backward()
                Doptimizer.step()


                print('[Discriminator training]')

                if train_step % mean_print == 0:
                    inputvis = vis.images((input[0:100, :, :, :] / 2 + 0.5).cpu(), win='inputvis', opts=dict(title='input'))
                    Goutputvis = vis.images((G_result1[0:100, :].view(-1, 3, target_size, target_size) / 2 + 0.5).cpu(), win='outputvis',
                                            opts=dict(title='Goutput1'))

                if train_step % 1 == 0:

                    G.zero_grad()

                    y = Variable(torch.zeros(current_batch_size).cuda())

                    v2 = Variable(torch.rand(current_batch_size, 100) * 2 - 1)
                    G_result2 = G(v2)

                    D_result = D(G_result2)
                    G_train_loss = - Loss(D_result, y)
                    G_train_loss.backward()
                    Goptimizer.step()

                    print('[Generator training]')
                    if train_step % mean_print == 0:
                        inputvis = vis.images((input[0:100, :, :, :] / 2 + 0.5).cpu(), win='inputvis', opts=dict(title='input'))
                        Goutputvis1 = vis.images((G_result2[0:100, :].view(-1, 3, target_size, target_size) / 2 + 0.5).cpu(), win='outputvis1',
                                                 opts=dict(title='Goutput2'))

                        DV = torch.Tensor([D_train_loss.item()])
                        lossvisD = vis.line(Y= DV, X = np.array([train_step]),win='lossvisD', update='append', opts=dict(title='Loss_D'))
                        rD = vis.line(Y=torch.Tensor([D_real_loss.item()]), X=np.array([train_step]), win='rD', update='append',opts=dict(title='D real loss'))
                        fD = vis.line(Y=torch.Tensor([D_fake_loss.item()]), X=np.array([train_step]), win='fD',update='append', opts=dict(title='D fake loss'))

        print('epoch{}'.format(epoch + 1))

    print('####### ### #     # ###  #####  #     # ####### ######     ### ### \n#        #  ##    #  #  #     # #     # #       #     #    ### ###\n#        #  # #   #  #  #       #     # #       #     #    ### ###\n#####    #  #  #  #  #   #####  ####### #####   #     #     #   #\n#        #  #   # #  #        # #     # #       #     #\n#        #  #    ##  #  #     # #     # #       #     #    ### ###\n#       ### #     # ###  #####  #     # ####### ######     ### ###')
