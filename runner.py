import torch
from torch import nn
from torch import optim
import os
from tqdm import tqdm
import glob

from tensorboardX import SummaryWriter

from model import ShapeAutoEncoder
from scheduler import WarmUpScheduler
from loader import load_dataset
from loss import RegularizeLoss, Assert
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
    net.train()
    for i in range(last_epoch+1, args.epoch):
        avg_loss = []
        for surfaces, queries, occupancies, images in tqdm(train_loader):
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
            Assert.check([out, gt])

            # write to tensorboard
            iou = metric_outs['iou']
            prec, reca = metric_outs['pr']
            Assert.check([iou])
            Assert.check([prec])
            Assert.check([reca])
            summary_writer.add_scalars('loss', dict(train_loss=loss), global_step)
            summary_writer.add_scalars('iou', dict(train_iou=iou), global_step)
            summary_writer.add_scalars('pr', dict(train_prec=prec, train_reca=reca), global_step)

            # record
            avg_loss.append(loss.item())
            global_step += 1

        warmUpScheduler.step()

        if (i + 1) % args.test_time == 0:
            net.eval()
            valid_loss = []
            valid_iou = []
            valid_prec, valid_reca = [], []

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
                valid_iou.append(iou)
                valid_prec.append(prec)
                valid_reca.append(reca)

            summary_writer.add_scalars('loss', dict(valid_loss=torch.mean(valid_loss)), global_step)
            summary_writer.add_scalars('iou', dict(valid_iou=torch.mean(valid_iou)), global_step)
            summary_writer.add_scalars('pr', dict(train_prec=torch.mean(valid_prec), train_reca=torch.mean(valid_reca)), global_step)

            avg_loss_ = torch.mean(avg_loss)
            tqdm.write(f'Average Loss During Last {args.test_times: d} Epoch is {avg_loss_: .6f}')

            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(net.state_dict(), f'{args.ckpt_dir}/{i + 1: d}.pt')

    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save(net.state_dict(), f'{args.ckpt_dir}/{args.epoch: d}.pt')


def eval(args):
    pass
