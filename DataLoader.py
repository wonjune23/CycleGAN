import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2 # if you have ROS installed in your system, this might raise error. Try deleting ROS from the system path.

# Train Dataset. It provides the data loader with images of both datasets (A and B) simultaneously.
class CycleGANDataset(torch.utils.data.Dataset):
    def __init__(self, args, transform):

        root = f'./datasets/{args.dataset}'
        if not os.path.isdir(root): # If the dataset is not present, download the dataset.
            print(f"\ndataset {args.dataset} doesn't exist: start downloading the dataset\n")
            os.system(f"bash ./datasets/download_cyclegan_dataset.sh {args.dataset}") # run the download script.
            print(f"\ndataset {args.dataset} download complete : start network {args.mode}ing now.\n")
        self.root = os.path.expanduser(root+'/'+args.mode) # Here, args.mode is always 'train'.
        self.transform = transform # preprocess.
        self.imagesNameA = os.listdir(self.root + 'A') # list of train images of dataset A.
        self.imagesNameB = os.listdir(self.root + 'B') # list of train images of dataset B.
        self.target_size = args.target_size # Target size to resize the images.

    def __len__(self): # __len__ defines the number of images per one epoch.
        # One epoch of iteration is defined as iteration through the minimum number of images among the two datasets (A and B).

        # For example, if dataset A has 100 images while B has 150 images, then 50 images of B are ignored.
        # One epoch is defined as 100 iteration images of both datsets.

        return min(len(self.imagesNameA), len(self.imagesNameB))

    def __getitem__(self, idx):
        imgA = cv2.imread(self.root + 'A/' + self.imagesNameA[idx]) # read the image
        imgA = cv2.resize(imgA, dsize=(self.target_size,self.target_size), interpolation=cv2.INTER_CUBIC) # resize the image
        imgA = np.array(imgA) # convert into numpy array
        imgA = self.transform(imgA) # convert into pytorch tensor, and then apply the preprocess(normalization)
        imgA = imgA[[2,1,0],:,:] # cv2 reads image as BGR. converting BGR into RGB here.

        imgB = cv2.imread(self.root + 'B/' + self.imagesNameB[idx])
        imgB = cv2.resize(imgB, dsize=(self.target_size, self.target_size), interpolation=cv2.INTER_CUBIC)
        imgB = np.array(imgB)
        imgB = self.transform(imgB)
        imgB = imgB[[2, 1, 0], :, :]

        return imgA, imgB # return images of A and B simultaneously.

# Test Dataset. It provides the data loader with one image at one time.
class CycleGANTestDataset(torch.utils.data.Dataset):
    def __init__(self, args, direction, transform):

        root = f'./datasets/{args.dataset}'
        if not os.path.isdir(root):
            print(f"\ndataset {args.dataset} doesn't exist: start downloading the dataset\n")
            os.system(f"bash ./datasets/download_cyclegan_dataset.sh {args.dataset}")
            print(f"\ndataset {args.dataset} download complete : start network {args.mode}ing now.\n")
        root = f'./datasets/{args.dataset}'

        self.root = os.path.expanduser(root+'/test')
        self.transform = transform
        self.data = 'A' if direction=='A2B' else 'B' # if the direction is A2B, we need to input images of A.
        self.imagesName = os.listdir(self.root + self.data) # list of images in the dataset (either A or B)

    def __len__(self):
        return len(self.imagesName)

    def __getitem__(self, idx):
        img = cv2.imread(self.root + f'{self.data}/' + self.imagesName[idx])
        img = np.array(img)
        img = self.transform(img)
        img = img[[2,1,0],:,:]

        return img, self.imagesName[idx] # we input the image with its name to match the name with its output.