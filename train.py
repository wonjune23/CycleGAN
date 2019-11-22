from models import *
from tqdm import tqdm
from DataLoader import CycleGANDataset

def train(args):
    if args.use_wandb:
        import wandb

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    e_identity = args.e_identity
    e_cycle = args.e_cycle

    if args.use_wandb:
        wandb.init(project="cycleGAN", name = f"{args.dataset}")

    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    G_A2B_optim = torch.optim.Adam(G_A2B.parameters(), lr = args.lr)
    G_B2A_optim = torch.optim.Adam(G_B2A.parameters(), lr=args.lr)
    D_A_optim = torch.optim.Adam(D_A.parameters(), lr = args.lr)
    D_B_optim = torch.optim.Adam(D_B.parameters(), lr=args.lr)

    L2Loss = nn.MSELoss()
    L1Loss = nn.L1Loss()
    train_step = 0

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CycleGANDataset(args, transform=transform)
    dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print(f'training started!\n max_epochs: {args.epoch}')
    for epoch in tqdm(range(args.epoch)):
        for i, (imgA, imgB) in enumerate(dataloader):

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
                    # Generator step

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

                    if args.use_wandb:
                        if train_step % 100 == 0:
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

    ckpt = {'G_A2B':G_A2B.state_dict(),
            'G_B2A':G_B2A.state_dict()}
    if not os.path.isdir('./checkpoints'):
        os.makedirs('./checkpoints')
    torch.save(ckpt, f'./checkpoints/{args.dataset}')
    print('\nmodels saved!')
    print('training finished!')
