import torch
from torch import nn
from torch import optim
import numpy as np

from network import SAE
from scheduler import WarmUpScheduler
from loss import RegularizeLoss

import argparse
from tqdm import tqdm, trange
from matplotlib import pyplot as plt


# parse argument
def parse():
    parser = argparse.ArgumentParser('3D-Generation')

    # model
    parser.add_argument('-num_features', type=int, default=128, help='number of latent features')
    parser.add_argument('-num_channels', type=int, default=512, help='number of channels')
    parser.add_argument('-num_layers', type=int, default=8, help='number of layers in decoder')
    parser.add_argument('-use_reg', type=bool, default=True, help='if use kl-regularization block')

    # training
    parser.add_argument('-learning_rate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('-weight_decay', type=float, default=1e-4, help='weight decay rate')
    parser.add_argument('-epoch', type=int, default=1000, help='number of training epoch')

    # optimizer
    parser.add_argument('-multiplier', type=float, default=2, help='learning rate warm up rate')

    # log
    parser.add_argument('-test_times', type=int, default=100, help='interval to test validation set')

    # test
    parser.add_argument('-threshold', type=float, default=0.3, help='threshold to set occupancy')

    args = parser.parse_args()
    return args


def main(args):
    N, B, M, D = 9600, 32, 1000, 250
    data = torch.rand((N, M, 3))
    net = SAE(args.num_features, args.num_channels, args.num_layers, args.use_reg).cuda()

    # gradually warm up opimization
    # SGDR: Stochastic Gradient Descent with Warm Restarts
    # https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Train.py
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = WarmUpScheduler(
        optimizer=optimizer, multiplier=args.multiplier, warm_epoch=args.epoch // 10, after_scheduler=cosineScheduler)

    bceloss = nn.BCELoss()
    regloss = RegularizeLoss()
    avg_loss = 0
    log = []
    
    # start training
    for i in trange(args.epoch):
        avg_loss = 0
        for j in range(N // B):
            optimizer.zero_grad()
            x = data[(j * B):((j + 1) * B)]
            y = torch.rand_like(x)
            # if near x then label 1 else label 0
            lab_y = (((x - y) ** 2).sum(dim=2) < 1e-5).float()
            
            # select D positive & D negative samples
            idx1 = torch.tensor([[i] * D for i in range(B)])
            idx2 = torch.tensor(np.array([np.random.choice(M, D, replace=False) for _ in range(B)]))
            inp = x[idx1, idx2]
            pnts = torch.cat((inp, y[idx1, idx2]), dim=1)
            labs = torch.cat((torch.ones((B, D)), lab_y[idx1, idx2]), dim=1)
            assert list(pnts.shape) == [B, D * 2, 3] and list(labs.shape) == [B, D * 2]

            # random shuffle
            idx1 = torch.tensor([[i] * (D * 2) for i in range(B)])
            idx2 = torch.tensor(np.array([np.random.choice(D * 2, D * 2, replace=False) for _ in range(B)]))
            pnts = pnts[idx1, idx2]
            labs = labs[idx1, idx2]
            assert list(pnts.shape) == [B, D * 2, 3] and list(labs.shape) == [B, D * 2]

            res = net(inp, pnts)
            loss_reg = regloss(res['regularize_mu'], res['regularize_var'])
            loss_bce = bceloss(res['occupancy'], labs.cuda())
            loss = loss_bce + loss_reg * 0.001
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        warmUpScheduler.step()
        log.append(avg_loss / (N // B))
        if (i + 1) % args.test_times == 0:
            tqdm.write(f'Average Loss During Last {args.test_times: d} Epoch is {avg_loss: .6f}')

    plt.plot(range(args.epoch), log)
    plt.title('Training Loss Curve')
    plt.savefig('./curves/loss.png')

    # start testing
    acc = 0
    for i in range(N // B):
        x = data[(i * B):((i + 1) * B)]
        res = net(x, x)
        acc += (res['occupancy'] > args.threshold).sum() / (B * M)
    print(f'Final Accuracy {acc / (N // B): .4f}')

    torch.save(net.state_dict(), f'./checkpoints/{args.epoch: d}.pt')


if __name__ == '__main__':
    main(parse())
