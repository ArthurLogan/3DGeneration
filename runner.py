import torch
from torch import nn
from torch import optim
import numpy as np
import os
from tqdm import tqdm
import glob
import time

from tensorboardX import SummaryWriter

from model import ShapeAutoEncoder, ShapeDenoiser
from diffusion import DiffusionTrainer
from scheduler import WarmUpScheduler
from loader import load_dataset
from loss import RegularizeLoss
from metric import Metric
from checkpoints import CheckpointIO


def train(args):
    """train autoencoder & denoiser"""
    train_autoencoder(args)
    # train_denoiser(args)


def train_autoencoder(args):
    # device
    torch.cuda.set_device(args['device'])

    # dataloader
    train_data, train_loader = load_dataset(args, mode='train', batch_size=args['train']['batch_size'])
    valid_data, valid_loader = load_dataset(args, mode='val', batch_size=32)
    print(f'train {len(train_data)}, valid {len(valid_data)}')

    # autoencoder
    model = ShapeAutoEncoder(
        features=args['autoencoder']['num_features'],
        channels=args['autoencoder']['num_channels'],
        layers=args['autoencoder']['num_layers'],
        reg_channels=args['denoiser']['num_channels']
    ).cuda()

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args['train']['learning_rate'], weight_decay=args['train']['weight_decay'])
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args['train']['epoch'], eta_min=0, last_epoch=-1)
    warmUpScheduler = WarmUpScheduler(
        optimizer=optimizer, multiplier=args['train']['multiplier'],
        warm_epoch=args['train']['epoch'] // 20, after_scheduler=cosineScheduler)
    
    # checkpoints
    checkpointIO = CheckpointIO(
        checkpoint_dir=args['log']['ckpt_dir'],
        model=model,
        scheduler=warmUpScheduler)
    scalars = dict()
    if args['autoencoder']['pretrained']:
        scalars = checkpointIO.load(args['autoencoder']['model_name'])

    # loss
    bceloss = nn.BCELoss().cuda()
    regloss = RegularizeLoss().cuda()
    criterion = lambda out, gt, mu, logvar: bceloss(out, gt) + regloss(mu, logvar) * args['train']['regularize_ratio']

    # summary writer
    dirs = glob.glob(f"{args['log']['log_dir']}/AE_*")
    os.makedirs(args['log']['log_dir'], exist_ok=True)
    summary_writer = SummaryWriter(f"{args['log']['log_dir']}/AE_{len(dirs)}")

    # process
    last_epoch = scalars.get('last_epoch', -1)
    best_loss = scalars.get('best_loss', np.inf)
    
    # start training
    for epoch in range(last_epoch+1, args['train']['epoch']):

        train_loss, train_iou, train_prec, train_reca = train_ae_step(train_loader, model, criterion, optimizer, epoch, args)
        valid_loss, valid_iou, valid_prec, valid_reca = valid_ae_step(valid_loader, model, criterion, args)

        warmUpScheduler.step()

        summary_writer.add_scalars('loss', dict(train_loss=train_loss, valid_loss=valid_loss), epoch)
        summary_writer.add_scalars('iou', dict(train_iou=train_iou, valid_iou=valid_iou), epoch)
        summary_writer.add_scalars('pr', dict(train_prec=train_prec, train_reca=train_reca,
                                              valid_prec=valid_prec, valid_reca=valid_reca), epoch)
        
        print(f'Average Loss In Validation Set During Last Epoch is {valid_loss: .6f}')
        checkpointIO.save('ae_model', last_epoch=epoch, best_loss=valid_loss)
        if valid_loss < best_loss:
            checkpointIO.save('best_ae_model', last_epoch=epoch, best_loss=valid_loss)
            best_loss = valid_loss

    summary_writer.close()


def train_ae_step(train_loader, model, criterion, optimizer, epoch, args):
    """train for one epoch"""
    print(f'Process Epoch {epoch}')

    model.train()

    train_loss = []
    train_iou = []
    train_prec, train_reca = [], []

    data_time, process_time = 0, 0

    end = time.time()
    for surfaces, queries, occupancies in tqdm(train_loader):

        # record data time
        data_time += time.time() - end
        end = time.time()

        # data transport
        surfaces = surfaces.cuda()
        queries = queries.cuda()
        occupancies = occupancies.cuda()

        # forward
        res = model(surfaces, queries)

        # loss
        loss = criterion(
            out=res['occupancy'],
            gt=occupancies,
            mu=res['regularize_mu'],
            logvar=res['regularize_var']
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['train']['grad_clip'])
        optimizer.step()

        # metric
        pred = (res['occupancy'] > args['test']['threshold']).int()
        gt = occupancies.int()
        metrics = Metric.get(pred, gt, metrics=['iou', 'pr'])
        iou = metrics['iou']
        prec, reca = metrics['pr']

        # record
        train_loss.append(loss.item())
        train_iou.append(iou.item())
        train_prec.append(prec.item())
        train_reca.append(reca.item())

        # record process time
        process_time += time.time() - end
        end = time.time()

    print(f'Data Time {data_time: .4f}, Process Time {process_time :.4f}')
    return np.mean(train_loss), np.mean(train_iou), np.mean(train_prec), np.mean(train_reca)


def valid_ae_step(valid_loader, model, criterion, args):
    """validate every epoch"""
    model.eval()

    valid_loss = []
    valid_iou = []
    valid_prec, valid_reca = [], []

    with torch.no_grad():
        for surfaces, queries, occupancies in tqdm(valid_loader):

            # data transport
            surfaces = surfaces.cuda()
            queries = queries.cuda()
            occupancies = occupancies.cuda()

            # forward
            res = model(surfaces, queries)

            # loss
            loss = criterion(
                out=res['occupancy'],
                gt=occupancies,
                mu=res['regularize_mu'],
                logvar=res['regularize_var']
            )

            # metric
            pred = (res['occupancy'] > args['test']['threshold']).int()
            gt = occupancies.int()
            metrics = Metric.get(pred, gt, metrics=['iou', 'pr'])
            iou = metrics['iou']
            prec, reca = metrics['pr']

            # record
            valid_loss.append(loss.item())
            valid_iou.append(iou.item())
            valid_prec.append(prec.item())
            valid_reca.append(reca.item())

    return np.mean(valid_loss), np.mean(valid_iou), np.mean(valid_prec), np.mean(valid_reca)


def train_denoiser(args):
    # device
    torch.cuda.set_device(args['device'])

    # dataloader
    train_data, train_loader = load_dataset(args, mode='train', batch_size=args['diffusion']['batch_size'])
    valid_data, valid_loader = load_dataset(args, mode='val', batch_size=32)
    print(f'train {len(train_data)}, valid {len(valid_data)}')

    # autoencoder & denoiser
    encoder = ShapeAutoEncoder(
        features=args['autoencoder']['num_features'],
        channels=args['autoencoder']['num_channels'],
        layers=args['autoencoder']['num_layers'],
        reg_channels=args['denoiser']['num_channels']
    ).cuda()
    denoiser = ShapeDenoiser(
        channels=args['denoiser']['num_channels'],
        layers=args['denoiser']['num_layers']
    ).cuda()
    model = DiffusionTrainer(
        model=denoiser,
        beta_1=args['diffusion']['beta_1'],
        beta_T=args['diffusion']['beta_T'],
        T=args['diffusion']['T']
    ).cuda()

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args['diffusion']['learning_rate'], weight_decay=args['diffusion']['weight_decay'])
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args['diffusion']['epoch'], eta_min=0, last_epoch=-1)
    warmUpScheduler = WarmUpScheduler(
        optimizer=optimizer, multiplier=args['diffusion']['multiplier'],
        warm_epoch=args['diffusion']['epoch'] // 20, after_scheduler=cosineScheduler)
    
    # load encoder & denoiser
    encoderIO = CheckpointIO(args['log']['ckpt_dir'], model=encoder)
    encoderIO.load(args['autoencoder']['model_name'])

    # freeze encoder
    for module in encoder.parameters():
        module.requires_grad = False

    scalars = dict()
    if args['denoiser']['pretrained']:
        checkpointIO = CheckpointIO(args['log']['ckpt_dir'], model=model, scheduler=warmUpScheduler)
        scalars = checkpointIO.load(args['log']['ckpt_dir'])

    # summary writer
    dirs = glob.glob(f"{args['log']['log_dir']}/Diff_*")
    os.makedirs(args['log']['log_dir'], exist_ok=True)
    summary_writer = SummaryWriter(f"{args['log']['log_dir']}/Diff_{len(dirs)}")

    # process
    last_epoch = scalars.get('last_epoch', -1)
    best_loss = scalars.get('best_loss', np.inf)

    # start training
    for epoch in range(last_epoch+1, args['train']['epoch']):

        train_loss = train_dm_step(train_loader, encoder, model, optimizer, epoch, args)
        valid_loss = valid_dm_step(valid_loader, encoder, model, args)

        warmUpScheduler.step()

        summary_writer.add_scalars('loss', dict(train_loss=train_loss, valid_loss=valid_loss), epoch)
        
        print(f'Average Loss In Validation Set During Last Epoch is {valid_loss: .6f}')
        checkpointIO.save('dm_model', last_epoch=epoch, best_loss=valid_loss)
        if valid_loss < best_loss:
            checkpointIO.save('best_dm_model', last_epoch=epoch, best_loss=valid_loss)
            best_loss = valid_loss

    summary_writer.close()


def train_dm_step(train_loader, embedder, model, optimizer, epoch, args):
    """train for one epoch"""
    print(f'Process Epoch {epoch}')

    model.train()

    train_loss = []
    data_time, process_time = 0, 0

    end = time.time()
    for surfaces, queries, occupancies in tqdm(train_loader):

        # record data time
        data_time += time.time() - end
        end = time.time()

        # data transport & forward
        surfaces = surfaces.cuda()
        feats, latents, mu, logvar = embedder.encode(surfaces)

        print(latents.shape)

        # loss
        loss = model(latents)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['diffusion']['grad_clip'])
        optimizer.step()

        # record
        train_loss.append(loss.item())

        # record process time
        process_time += time.time() - end
        end = time.time()

    print(f'Data Time {data_time: .4f}, Process Time {process_time :.4f}')
    return np.mean(train_loss)


def valid_dm_step(valid_loader, embedder, model, args):
    """validate every epoch"""
    model.eval()

    valid_loss = []
    with torch.no_grad():
        for surfaces, queries, occupancies in tqdm(valid_loader):

            # data transport & forward
            surfaces = surfaces.cuda()
            feats, latents, mu, logvar = embedder.encode(surfaces)

            # loss
            loss = model(latents)

            # record
            valid_loss.append(loss.item())

    return np.mean(valid_loss)

