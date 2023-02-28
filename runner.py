import torch
from torch import nn
from torch import optim
import numpy as np
import os
from tqdm import tqdm

from tensorboardX import SummaryWriter

from model import ShapeAutoEncoder
from scheduler import WarmUpScheduler
from loader import load_dataset
from loss import RegularizeLoss


def train(args):
    # device
    device = torch.device(args.device)

    # dataset
    dataset, dataloader = load_dataset(args)

    # model
    net = ShapeAutoEncoder(args.num_features, args.num_channels, args.num_layers, args.use_reg, device).to(device)

    # gradually warm up opimization
    # SGDR: Stochastic Gradient Descent with Warm Restarts
    # https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Train.py
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = WarmUpScheduler(
        optimizer=optimizer, multiplier=args.multiplier, warm_epoch=args.epoch // 10, after_scheduler=cosineScheduler)

    # loss 
    bceloss = nn.BCELoss()
    regloss = RegularizeLoss()
    avg_loss = 0
    log = []

    # summary writer
    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = SummaryWriter(args.log_dir)

    # process var
    last_epoch = -1
    global_step = 0
    
    # start training
    for i in range(last_epoch+1, args.epoch):
        avg_loss = 0
        local_step = 0
        with tqdm(enumerate(dataloader)) as tqdmLoader:
            for j, (positions, occupancies, images) in tqdmLoader:
                optimizer.zero_grad()
                res = net(positions, positions)

                occupancies = occupancies.to(device)

                loss_reg = regloss(res['regularize_mu'], res['regularize_var'])
                loss_bce = bceloss(res['occupancy'], occupancies)
                loss = loss_bce + loss_reg * 0.001
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()

                occupied = (res['occupancy'] > args.threshold).int()
                occupancy_gt = occupancies.int()
                precision = (occupied == occupancy_gt).sum() / occupied.sum()
                recall = (occupied == occupancy_gt).sum() / occupancy_gt.sum()

                summary_writer.add_scalars(
                    main_tag='loss',
                    tag_scalar_dict=dict(training=loss),
                    global_step=global_step
                )

                summary_writer.add_scalars(
                    main_tag='precision',
                    tag_scalar_dict=dict(training=precision),
                    global_step=global_step
                )

                summary_writer.add_scalars(
                    main_tag='recall',
                    tag_scalar_dict=dict(training=recall),
                    global_step=global_step
                )

                tqdmLoader.set_postfix(ordered_dict={
                    "epoch": i,
                    "global_step": global_step,
                    "loss": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

                local_step += 1
                global_step += 1

            warmUpScheduler.step()
        
        avg_loss /= local_step
        if (i + 1) % args.test_times == 0:
            tqdm.write(f'Average Loss During Last {args.test_times: d} Epoch is {avg_loss: .6f}')

    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save(net.state_dict(), f'{args.ckpt_dir}/{args.epoch: d}.pt')


def eval(args):
    pass
