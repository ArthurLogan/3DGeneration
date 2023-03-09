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

# distributed data parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# set up process
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


# clean up process
def cleanup():
    dist.destroy_process_group()


# check main proc
def is_main_proc():
    return dist.get_global_rank() == 0


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


def train_ddp(args):
    # ddp setup
    setup(args.local_rank, 4)

    # device
    device = torch.device(f'cuda:{args.local_rank}')
    # dataloader
    train_data, train_loader = load_dataset(args, mode='train')
    valid_data, valid_loader = load_dataset(args, mode='val')
    print(f"train {len(train_data)}, valid {len(valid_data)}")

    # model
    model = ShapeAutoEncoder(args.model.num_features, args.model.num_channels,
                             args.model.num_layers, args.model.regularized).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

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
    bceloss = nn.BCELoss().to(device)
    regloss = RegularizeLoss().to(device)

    # summary writer
    dirs = glob.glob(f"{args.training.log_dir}/*")
    os.makedirs(args.training.log_dir, exist_ok=True)
    summary_writer = None
    if is_main_proc():
        summary_writer = SummaryWriter(f"{args.training.log_dir}/{len(dirs)}")

    # process
    last_epoch = scalars.get('last_epoch', -1)
    global_step = scalars.get('global_step', 0)
    
    # start training
    for i in range(last_epoch+1, args.training.epoch):

        model.train()
        train_loader.sampler.set_epoch(i)

        avg_loss = []
        if is_main_proc():
            tqdmloader = tqdm(train_loader)
        else:
            tqdmloader = train_loader
        for surfaces, queries, occupancies, images in tqdmloader:
            optimizer.zero_grad()

            # forward
            res = model(surfaces, queries, device)
            
            # loss
            occupancies = occupancies.to(device)
            loss_reg = regloss(res['regularize_mu'], res['regularize_var'])
            loss_bce = bceloss(res['occupancy'], occupancies)
            loss = loss_bce + loss_reg * args.training.regularized_ratio
            loss.backward()
            optimizer.step()

            reduce_loss(loss, dist.get_rank(), 4)

            if is_main_proc():
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

                tqdmloader.set_postfix(dict(epoch=i, global_step=global_step))

            # record
            avg_loss.append(loss.item())
            global_step += 1

        warmUpScheduler.step()

        if (i + 1) % args.training.validate_every == 0 and is_main_proc():
            model.eval()
            valid_loss = []
            valid_iou = []
            valid_prec, valid_reca = [], []

            with torch.no_grad():
                for surfaces, queries, occupancies, images in tqdm(valid_loader):

                    # forward
                    res = model(surfaces, queries, device)

                    # loss
                    occupancies = occupancies.to(device)
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

    if is_main_proc():
        summary_writer.close()
        checkpointIO.save(f'{i+1: d}.pt', last_epoch=i, global_step=global_step)
