import os, sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer

def pretrain():
    T = Trainer()
    T.task = 'debug'
    T.note = f'debug'
    T.batch = 64
    T.epochs = 800
    T.warmup_epochs = 40
    T.input_size = 224
    T.accum_iter = 16
    T.device = '0,1,2,3'
    T.dataset = 'ImageNet-LT'
    T.model = f'mae_vit_base_patch16'
    T.mask_ratio = 0.75
    T.blr = 1.5e-4
    T.weight_decay = 0.05
    T.num_workers = 16
    T.pretrain()

pretrain()