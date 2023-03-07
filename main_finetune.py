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

import timm
assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loss import *
from models import vit

from engine_finetune import train_one_epoch, evaluate
from engine_finetune import evaluate_all_metric

import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')

    # * Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # * Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--adamW2', type=float, default=0.95)

    # * Learning rate parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # * Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * loss params
    parser.add_argument('--loss', type=str, default='ce', 
                        help='loss type')
    parser.add_argument('--bal_tau', type=float, default=1.0, 
                        help='margin factor of BalCE or BalBCE')

    # * Finetuning params
    parser.add_argument('--finetune', default='', 
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--attn_only', action='store_true')
    parser.set_defaults(attn_only=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # * Dataset parameters
    parser.add_argument('--data_path', default='/diskC/xzz/ImageNet-LT', type=str, 
                        help='dataset root path')
    parser.add_argument('--nb_classes', default=1000, type=int, 
                        help='number of the classification types')
    parser.add_argument('--dataset', default='ImageNet-LT', type=str, 
                        help='dataset name')
    parser.add_argument('--imbf', default=100, type=int, 
                        help='imbalance factor, only required for CIFAR')

    # * File parameters
    parser.add_argument('--ckpt_dir', default='./ckpt_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed fixed for reproducing')
    parser.add_argument('--prit', default=20, type=int,
                        help='console info print frequency')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    # * Load parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', 
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', 
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # * distributed training parameters
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
    P.debug = True if args.eval else P.flush()
    P.log('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    P.log("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        P.log("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                P.log('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    args.cls_num = dataset_train.get_cls_num()
    P.log(f'Train on {dataset_train.__len__()} Image w.r.t. {len(args.cls_num)} classes')

    model = vit.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        P.log("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                P.log(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        P.log(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)


    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    P.log(f"Model = {args.model}")
    P.log('number of params (M): %.2f' % (n_parameters / 1.e6))
     
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    P.log("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    P.log("actual lr: %.2e" % args.lr)

    P.log("accumulate grad iterations: %d" % args.accum_iter)
    P.log("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, args.adamW2))

    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, 
                    optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        result, cf = evaluate_all_metric(data_loader_val, model, device, args)
        save_pth = args.resume.strip('cheakpoint.pth')
        misc.save_eval_json(result, save_pth)
        np.save(save_pth + "cf.npy", cf, allow_pickle=True)
        P.log(f"Save all results @ {save_pth}")
        exit(0)
        
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        P.log("Mixup is activated!")
        mixup_fn = Mixup(mixup_alpha=args.mixup, 
                        cutmix_alpha=args.cutmix, 
                        cutmix_minmax=args.cutmix_minmax,
                        prob=args.mixup_prob, 
                        switch_prob=args.mixup_switch_prob, 
                        mode=args.mixup_mode,
                        label_smoothing=args.smoothing, 
                        num_classes=args.nb_classes)

    if mixup_fn is not None:
        if args.loss == 'CE': criterion = ST_CE_loss()
        elif args.loss == 'Bal_CE': criterion = Bal_CE_loss(args)
        elif args.loss == 'BCE': criterion = BCE_loss()
        elif args.loss == 'CB_BCE': criterion = BCE_loss(args, type='CB')
        elif args.loss == 'Bal_BCE': criterion = BCE_loss(args, type='Bal')
        elif args.loss == 'MiSLAS': criterion = MiSLAS_loss(args)
        elif args.loss == 'LDAM': criterion = LDAM_loss(args)
    else:
        if args.loss == 'CE': criterion = torch.nn.CrossEntropyLoss()
        elif args.loss == 'LS_CE': criterion = LS_CE_loss(smoothing=args.smoothing)
        elif args.loss == 'CB_CE': criterion = CB_CE_loss(args)
        elif args.loss == 'LADE': criterion = LADE_loss(args)
        else: raise NotImplementedError("No loss setting!")

    P.log("criterion = %s" % str(criterion))

    if not args.eval: misc.save_args_txt(args)


    P.log(f"Start training for {args.epochs} epochs")
    P.timing(reset=True)
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        if epoch % args.prit == 0 or epoch + 1 == args.epochs:
            test_stats = evaluate(data_loader_val, model, device)
            P.log(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            
            if test_stats["acc1"] > max_accuracy and args.ckpt_dir:
                misc.save_model(args=args, model=model, 
                model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

            max_accuracy = max(max_accuracy, test_stats["acc1"])
            P.log(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
        else: # not eval to save time
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

        if args.log_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            P.log(json.dumps(log_stats))
        time_one_epoch = P.timing()
        P.log(f'Training epoch {epoch} for {time_one_epoch}')

    misc.save_args_txt(args, max_accuracy)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    P.log('Total training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.ckpt_dir and not args.eval:
        Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir and not args.eval:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)
