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
    testset = CycleGANTestDataset(args, direction='A2B', transform=transform)
    dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    print(f'A2B conversion started!\n')
    for img, name in tqdm(dataloader):

        realA = img.to(device)
        fakeB = G_A2B(realA)
        result = (fakeB[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy()) / 2 + 0.5
        plt.imsave(f'./results/{args.dataset}/A2B/{name[0]}', result)


    testset = CycleGANTestDataset(args, direction='B2A', transform=transform)
    dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    print(f'B2A conversion started!\n')
    for img, name in tqdm(dataloader):

        realB = img.to(device)
        fakeA = G_B2A(realB)
        result = (fakeA[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy()) / 2 +0.5
        plt.imsave(f'./results/{args.dataset}/B2A/{name[0]}', result)

    print(f'{args.dataset} testing finished!')
