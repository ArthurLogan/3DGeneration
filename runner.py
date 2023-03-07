import torch
from torch import nn
from torch import optim
import numpy as np
import os
from tqdm import tqdm
import glob

from tensorboardX import SummaryWriter

from model import ShapeAutoEncoder
from scheduler import WarmUpScheduler
from loader import load_dataset
from loss import RegularizeLoss
from metric import Metric


def train(args):
    # device
    device = torch.device(args.device)

    # dataloader
    train_data, train_loader = load_dataset(args, mode='train')
    valid_data, valid_loader = load_dataset(args, mode='val')
    print(f"train {len(train_data)}, valid {len(valid_data)}")

    # model
    net = ShapeAutoEncoder(args.num_features, args.num_channels, args.num_layers, args.use_reg).to(device)

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
    bceloss = nn.BCELoss().to(device)
    regloss = RegularizeLoss().to(device)

    # summary writer
    dirs = glob.glob(f"{args.log_dir}/*")
    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = SummaryWriter(f"{args.log_dir}/{len(dirs)}")

    # process
    last_epoch = -1
    global_step = 0
    
    # start training
    for i in range(last_epoch+1, args.epoch):

        net.train()

        avg_loss = []
        tqdmloader = tqdm(train_loader)
        for surfaces, queries, occupancies, images in tqdmloader:
            optimizer.zero_grad()

            # forward
            res = net(surfaces, queries, device)
            
            # loss
            occupancies = occupancies.to(device)
            loss_reg = regloss(res['regularize_mu'], res['regularize_var'])
            loss_bce = bceloss(res['occupancy'], occupancies)
            loss = loss_bce + loss_reg * args.reg_ratio
            loss.backward()
            optimizer.step()

            # metric
            out = (res['occupancy'] > args.threshold).int()
            gt = occupancies.int()
            metric_outs = Metric.get(out, gt, metrics=['iou', 'pr'])

            # write to tensorboard
            iou = metric_outs['iou']
            prec, reca = metric_outs['pr']
            summary_writer.add_scalars('loss', dict(train_loss=loss.item()), global_step)
            summary_writer.add_scalars('iou', dict(train_iou=iou.item()), global_step)
            summary_writer.add_scalars('pr', dict(train_prec=prec.item(), train_reca=reca.item()), global_step)

            # record
            avg_loss.append(loss.item())
            global_step += 1
            tqdmloader.set_postfix(dict(epoch=i, global_step=global_step))

        warmUpScheduler.step()

        if (i + 1) % args.test_time == 0:
            net.eval()
            valid_loss = []
            valid_iou = []
            valid_prec, valid_reca = [], []

            with torch.no_grad():
                for surfaces, queries, occupancies, images in tqdm(valid_loader):

                    # forward
                    res = net(surfaces, queries, device)

                    # loss
                    occupancies = occupancies.to(device)
                    loss_reg = regloss(res['regularize_mu'], res['regularize_var'])
                    loss_bce = bceloss(res['occupancy'], occupancies)
                    loss = loss_bce + loss_reg * args.reg_ratio

                    # metric
                    out = (res['occupancy'] > args.threshold).int()
                    gt = occupancies.int()
                    metric_outs = Metric.get(out, gt, metrics=['iou', 'pr'])

                    # record
                    iou = metric_outs['iou']
                    prec, reca = metric_outs['pr']
                    valid_loss.append(loss.item())
                    valid_iou.append(iou.item())
                    valid_prec.append(prec.item())
                    valid_reca.append(reca.item())

            summary_writer.add_scalars('loss', dict(valid_loss=np.mean(valid_loss)), global_step)
            summary_writer.add_scalars('iou', dict(valid_iou=np.mean(valid_iou)), global_step)
            summary_writer.add_scalars('pr', dict(train_prec=np.mean(valid_prec), train_reca=np.mean(valid_reca)), global_step)

            avg_loss_ = np.mean(avg_loss)
            tqdm.write(f'Average Loss During Last {args.test_time: d} Epoch is {avg_loss_: .6f}')

            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(net.state_dict(), f'{args.ckpt_dir}/{i + 1: d}.pt')

    summary_writer.close()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save(net.state_dict(), f'{args.ckpt_dir}/{args.epoch: d}.pt')


def eval(args):
    pass
