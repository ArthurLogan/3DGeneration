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
from utils.checkpoints import CheckpointIO

# torch lightning training
from pytorch_lightning.lite import LightningLite


def run(args):
    # dataloader
    train_data, train_loader = load_dataset(args, mode='train')
    valid_data, valid_loader = load_dataset(args, mode='val')
    print(f"train {len(train_data)}, valid {len(valid_data)}")

    # model
    model = ShapeAutoEncoder(args.model.num_features, args.model.num_channels,
                             args.model.num_layers, args.model.regularized)

    # gradually warm up opimization
    # SGDR: Stochastic Gradient Descent with Warm Restarts
    # https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Train.py
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.training.learning_rate, weight_decay=args.training.weight_decay)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.training.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = WarmUpScheduler(
        optimizer=optimizer, multiplier=args.training.multiplier, warm_epoch=args.training.epoch // 10, after_scheduler=cosineScheduler)
    
    # checkpoints
    checkpointIO = CheckpointIO(checkpoint_dir=args.training.ckpt_dir, model=model, scheduler=warmUpScheduler)
    scalars = dict()
    if args.training.pretrained:
        scalars = checkpointIO.load(args.training.ckpt_name)

    # loss
    bceloss = nn.BCELoss()
    regloss = RegularizeLoss()

    # summary writer
    dirs = glob.glob(f"{args.training.log_dir}/*")
    os.makedirs(args.training.log_dir, exist_ok=True)
    summary_writer = SummaryWriter(f"{args.training.log_dir}/{len(dirs)}")

    # process
    last_epoch = scalars.get('last_epoch', -1)
    global_step = scalars.get('global_step', 0)
    
    # start training
    for i in range(last_epoch+1, args.training.epoch):

        model.train()
        train_loader.sampler.set_epoch(i)

        avg_loss = []
        tqdmloader = tqdm(train_loader)
        for surfaces, queries, occupancies in tqdmloader:
            optimizer.zero_grad()

            # forward
            res = model(surfaces, queries)
            
            # loss
            loss_reg = regloss(res['regularize_mu'], res['regularize_var'])
            loss_bce = bceloss(res['occupancy'], occupancies)
            loss = loss_bce + loss_reg * args.training.regularized_ratio
            loss.backward()
            optimizer.step()

            # metric
            out = (res['occupancy'] > args.test.threshold).int()
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

        if (i + 1) % args.training.validate_every == 0:
            model.eval()
            valid_loss = []
            valid_iou = []
            valid_prec, valid_reca = [], []

            with torch.no_grad():
                for surfaces, queries, occupancies in tqdm(valid_loader):

                    # forward
                    res = model(surfaces, queries)

                    # loss
                    loss_reg = regloss(res['regularize_mu'], res['regularize_var'])
                    loss_bce = bceloss(res['occupancy'], occupancies)
                    loss = loss_bce + loss_reg * args.training.regularized_ratio

                    # metric
                    out = (res['occupancy'] > args.test.threshold).int()
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
            tqdm.write(f'Average Loss During Last {args.training.validate_every: d} Epoch is {avg_loss_: .6f}')
            checkpointIO.save(f'{i+1: d}.pt', last_epoch=i, global_step=global_step)

    summary_writer.close()
    checkpointIO.save(f'{i+1: d}.pt', last_epoch=i, global_step=global_step)
