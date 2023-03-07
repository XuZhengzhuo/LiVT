import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH

def run_imagenetlt():
    T = Trainer()
    T.task = 'debug'
    T.note = 'debug'
    T.ckpt = '/home/hfx/LiVT/ckpt/pretrain/ImageNet-LT/vit_base_224/checkpoint.pth'

    T.dataset = 'ImageNet-LT'
    T.nb_classes = 1000

    T.epochs = 100
    T.device = '0,1,2,3'

    T.batch = 64
    T.accum_iter = 8

    T.model = 'vit_base_patch16'
    T.input_size = 128
    T.drop_path = 0.1

    T.clip_grad = None
    T.weight_decay = 0.05
    T.adamW2 = 0.999

    T.lr = None
    T.blr = 1e-3
    T.layer_decay = 0.65
    T.min_lr = 1e-6
    T.warmup_epochs = 5

    T.color_jitter = None
    T.aa = 'rand-m9-mstd0.5-inc1'

    T.reprob = 0.25
    T.remode = 'pixel'
    T.recount = 1
    T.resplit = False

    T.mixup = 0.
    T.cutmix = 0.
    T.cutmix_minmax = None
    T.mixup_prob = 1.0
    T.mixup_switch_prob = 0.5
    T.mixup_mode = 'batch'

    T.loss = 'Bal_BCE'
    T.bal_tau = 1.0
    T.smoothing = 0.1

    T.global_pool = True

    T.seed = 0
    T.prit = 1

    T.num_workers = 16
    T.master_port = 29500

    T.finetune()

run_imagenetlt()

