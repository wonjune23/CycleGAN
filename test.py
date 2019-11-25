from models import *
from tqdm import tqdm
from DataLoader import CycleGANTestDataset
import matplotlib.pyplot as plt

def test(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    G_A2B = Generator().to(device) # We only need the Generators for testing.
    G_B2A = Generator().to(device)

    ckpt = torch.load(f'./checkpoints/{args.dataset}') # loading the pretrained model.
    G_A2B.load_state_dict(ckpt['G_A2B'])
    G_B2A.load_state_dict(ckpt['G_B2A'])

    if not os.path.isdir(f'./results/{args.dataset}'): # if the directory 'results' is not present, make the directory.
        os.makedirs(f'./results/{args.dataset}/A2B')
        os.makedirs(f'./results/{args.dataset}/B2A')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # the preprocesses.
    testset = CycleGANTestDataset(args, direction='A2B', transform=transform) # test datset for A2B translation.
    dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2) # test data loader for A2B translation.

    print(f'A2B conversion started!\n')
    for img, name in tqdm(dataloader):

        real = img.to(device)
        fake = G_A2B(real) # output image.
        result = (fake[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy()) / 2 + 0.5 # undo the preprocess (denormalization).
        plt.imsave(f'./results/{args.dataset}/A2B/{name[0]}', result) # save the output.


    testset = CycleGANTestDataset(args, direction='B2A', transform=transform) # test datset for B2A translation.
    dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2) # test data loader for B2A translation.
    print(f'B2A conversion started!\n')
    for img, name in tqdm(dataloader):

        real = img.to(device)
        fake = G_B2A(real) # output image.
        result = (fake[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy()) / 2 +0.5 # undo the preprocess (denormalization).
        plt.imsave(f'./results/{args.dataset}/B2A/{name[0]}', result) # save the output.

    print(f'{args.dataset} testing finished!')
