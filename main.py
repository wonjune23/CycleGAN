import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import os
from train import train
from DataLoader import CycleGANDataset

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type = int, default = 0, help = 'GPU number to use')
parser.add_argument('--batch_size', type = int, default = 1, help = 'batch size')
parser.add_argument('--target_size', type = int, default = 128, help = 'image is resize to this value')
parser.add_argument('--num_resblock', type = int, default = 6, help = 'number of resblocks in generator')
parser.add_argument('--dataset', type = str, required=True, help = 'dataset to use. e.g., horse2zebra')
parser.add_argument('--mode', type = str, required=True, help = 'train / test')
parser.add_argument('--epoch', type = int, default = 200, help = 'number of epochs to train')
parser.add_argument('--use_wandb', type = int, default = 0, help = 'flag to use wandb or not')
parser.add_argument('--e_identity', type = float, default = 0, help = 'lambda for identity loss')
parser.add_argument('--e_cycle', type = float, default = 10, help = 'lambda for cycle loss')
parser.add_argument('--lr', type = float, default = 1e-4, help = 'learning rate')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu}"

batch_size = args.batch_size
target_size = args.target_size

if __name__  == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CycleGANDataset(args, transform=transform)
    dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.mode == 'train':
        train(args, dataloader)
    elif args.mode == 'test':
        pass
    else:
        raise NotImplementedError(f"args.mode takes only 'train' or 'test', but you have passed {args.mode}.")