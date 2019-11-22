from models import *
from tqdm import tqdm
from DataLoader import CycleGANTestDataset
import matplotlib.pyplot as plt

def test(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)

    ckpt = torch.load(f'./checkpoints/{args.dataset}')
    G_A2B.load_state_dict(ckpt['G_A2B'])
    G_B2A.load_state_dict(ckpt['G_B2A'])

    if not os.path.isdir(f'./results/{args.dataset}'):
        os.makedirs(f'./results/{args.dataset}/A2B')
        os.makedirs(f'./results/{args.dataset}/B2A')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = CycleGANTestDataset(direction='A2B', transform=transform)
    dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    print(f'A2B conversion started!\n')
    for i, (img, name) in tqdm(enumerate(dataloader)):

        realA = imgA.to(device)
        fakeB = G_A2B(realA)

        plt.imsave(f'./results/{args.dataset}/A2B/name',255*(fakeB.cpu().numpy()/2+0.5))

    testset = CycleGANTestDataset(direction='B2A', transform=transform)
    dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    print(f'B2A conversion started!\n')
    for i, (img, name) in tqdm(enumerate(dataloader)):

        realB = imgB.to(device)
        fakeA = G_B2A(realB)

        plt.imsave(f'./results/{args.dataset}/B2A/name', 255 * (fakeA.cpu().numpy() / 2 + 0.5))

    print('testing finished!')
