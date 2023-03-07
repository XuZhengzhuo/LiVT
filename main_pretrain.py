# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import DatasetLT

from models import mae

from engine_pretrain import train_one_epoch

import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')

    # * Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False) 

    # * Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # * Dataset parameters
    parser.add_argument('--dataset', default='ImageNet-LT', type=str,
                        help='dataset name')
    parser.add_argument('--data_path', default='/diskC/xzz/ImageNet-LT', type=str,
                        help='dataset path')

    # * File parameters
    parser.add_argument('--ckpt_dir', default='./ckpt',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./exp',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', 
                        help='resume from checkpoint')
    parser.add_argument('--prit', default=200, type=int,
                        help='console info print frequency')

    # * Load parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # * Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    P = misc.Printer(os.path.join(args.log_dir, "log.txt"))
    P.flush()
    P.log('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    P.log("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.empty_cache()
    cudnn.benchmark = True

    # simple augmentation
    if args.dataset == 'ImageNet-LT': # ImageNet-LT
        Img_Norm = transforms.Normalize(mean=[0.479672, 0.457713, 0.407721], std=[0.278976, 0.271203, 0.286062])
    elif args.dataset == 'iNat18': # iNat18
        Img_Norm = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
    elif args.dataset == 'ImageNet-BAL': # ImageNet-BAL
        Img_Norm = transforms.Normalize(mean=[0.480767, 0.457071, 0.407718], std=[0.279940, 0.272481, 0.286038])
    else: # ImageNet
        Img_Norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Img_Norm])
            
    dataset_train = DatasetLT(os.path.join(args.data_path, 'train'), transform=transform_train)
    P.log(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        P.log("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    P.log(f'log_writer is {log_writer}')
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    args.cls_num = dataset_train.get_cls_num()

    # define the model
    model = mae.__dict__[args.model](img_size=args.input_size, norm_pix_loss=args.norm_pix_loss)
    model.set_criterion(args)
    model.to(device)

    model_without_ddp = model
    # P.log("Model = %s" % str(model_without_ddp))
    P.log(f"Model = {args.model}")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    P.log("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    P.log("actual lr: %.2e" % args.lr)

    P.log("accumulate grad iterations: %d" % args.accum_iter)
    P.log("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    P.log(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    misc.save_args_txt(args)

    P.log(f"Start training for {args.epochs} epochs")
    P.timing(reset=True)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.ckpt_dir:
            misc.save_model(args=args, model=model, 
                            model_without_ddp=model_without_ddp, 
                            optimizer=optimizer,
                            loss_scaler=loss_scaler, 
                            epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.log_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            P.log(json.dumps(log_stats))
        time_one_epoch = P.timing()
        P.log(f'Training epoch {epoch} for {time_one_epoch}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    P.log('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.ckpt_dir:
        Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)
