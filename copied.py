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

os.environ["CUDA_VISIBLE_DEVICES"]="2"

batch_size = 128

target_size = 64

vis = visdom.Visdom( port='8097', env = 'DCGAN_3')


class PokemonDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.imagesName = os.listdir(self.root)

    def __len__(self):
        return len(self.imagesName)

    def __getitem__(self, idx):
        img = cv2.imread(self.root + '/' + self.imagesName[idx])
        img = cv2.resize(img, dsize=(target_size,target_size), interpolation=cv2.INTER_CUBIC)
        img = np.array(img)

        img = self.transform(img)
        #print(img.shape)
        img = img[[2,1,0],:,:]
        #print(img.shape)
        return img, idx

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = PokemonDataset(root = './train/train', transform = transform)
#estset = torchvision.datasets.MNIST(root = './MNIST/test', train = False, download = True, transform = transform)

trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 2)
#testloader = DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #self.fc1 = nn.Linear(100, 1024*4*4) # 4x4
        #self.fc2 = nn.BatchNorm2d(1024)
        #self.fc3 = nn.ReLU()

        self.f1 = nn.ConvTranspose2d(100, 1024, 4, 1, 0) # 4x4
        self.f2 = nn.BatchNorm2d(1024)
        self.f3 = nn.ReLU()

        self.c1 = nn.ConvTranspose2d(1024,512, 4, stride = 2, padding= 1) # 8x8
        self.c2 = nn.BatchNorm2d(512)
        self.c3 = nn.ReLU()
        self.c4 = nn.ConvTranspose2d(512,256, 4, 2, 1) # 16x16
        self.c5 = nn.BatchNorm2d(256)
        self.c6 = nn.ReLU()
        self.c7 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # 32x32
        self.c8 = nn.BatchNorm2d(128)
        self.c9 = nn.ReLU()
        self.c10 = nn.ConvTranspose2d(128, 3, 4, 2, 1)  # 64x64

    def forward(self, input):
        # input : B, 3, target_size, target_size
        #x = self.fc3(self.fc2((self.fc1(input)).reshape(-1,1024,4,4)))
        x = input.reshape(-1, 100, 1, 1)
        x = self.f1(x)
        #print(x.shape)
        x = self.f2(x)
        x = self.f3(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.c8(x)
        x = self.c9(x)
        x = self.c10(x)
        #print(x.shape)

        return F.tanh(x) # returns -1 ~ 1 output and latent vector

class Descriminator(nn.Module):
    def __init__(self):

        super(Descriminator, self).__init__()
        self.f1 = nn.Conv2d(3, 128, 4, stride=2, padding=1)  # 64x64
        self.f2 = nn.BatchNorm2d(128)
        self.f3 = nn.LeakyReLU()
        self.c1 = nn.Conv2d(128, 256, 4, stride = 2, padding = 1) # 32x32
        self.c2 = nn.BatchNorm2d(256)
        self.c3 = nn.LeakyReLU()
        self.c4 = nn.Conv2d(256, 512, 4, stride = 2, padding = 1) # 16x16
        self.c5 = nn.BatchNorm2d(512)
        self.c6 = nn.LeakyReLU()
        self.c7 = nn.Conv2d(512, 1024, 4, stride = 2, padding = 1) # 8x8
        self.c8 = nn.BatchNorm2d(1024)
        self.c9 = nn.LeakyReLU()
        self.c10 = nn.Conv2d(1024,1024, 4, stride = 2, padding = 1) # 4x4
        self.c11 = nn.BatchNorm2d(1024)
        self.c12 = nn.LeakyReLU()
        self.c13 = nn.Conv2d(1024, 1, 4, stride = 2, padding = 1) # 1x1
        #self.c14 = nn.BatchNorm2d(1024)
        #self.c15 = nn.LeakyReLU()
        #self.c16 = nn.Conv2d(1024, 1, 4, stride = 2, padding = 1) # 1x1

    def forward(self, input):
        #x = input.reshape(-1, 3, target_size, target_size)
        x = self.f1(input)
        x = self.f2(x)
        x = self.f3(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.c8(x)
        x = self.c9(x)
        x = self.c10(x)
        x = self.c11(x)
        x = self.c12(x)
        x = self.c13(x)

        #x = self.c14(x)
        #x = self.c15(x)

        #x = self.c16(x)

        x = x.squeeze()

        return F.sigmoid(x) # no sigmoid for WGAN

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
