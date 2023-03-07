import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())

DATA_PATH = '/home/hfx/DATA'
WORK_PATH = '/home/hfx/LiVT'

IMAGENET_LT_PATH = os.path.join(DATA_PATH, 'ImageNet-LT')
IMAGENET_BAL_PATH = os.path.join(DATA_PATH, 'ImageNet-BAL')
INAT18_PATH = os.path.join(DATA_PATH, 'iNat18')
PLCAE_PATH = os.path.join(DATA_PATH, 'Places_LT')
CIFAR10_PATH = os.path.join(DATA_PATH)
CIFAR100_PATH = os.path.join(DATA_PATH)
EXP_PATH = os.path.join(WORK_PATH, 'exp')


class Trainer():
    def __init__(self,) -> None:

        self.task = '' # e.g. pretrain
        self.note = '' # e.g. MGP_Epoch_800
        self.ckpt = '' # the fintune ckpt path
        
        self.dataset = 'ImageNet-LT' # keep consisitent with your dataset fold name.
        self.nb_classes = 1000 # e.g. ImageNet-LT 1000
        self.epochs = 800 # e.g. MGP 800 BFT 100
        
        self.batch = 256 # Adapt to your GPU Memory
        self.accum_iter = 4 # eff_batch = batch * GPUs * accum_iter
        self.device = '0,1,2,3'
        
        self.model = 'mae_vit_base_patch16' # different in MGP and BFT stage
        self.resume = '' 
        self.input_size = 224 # e.g. iNat at MGP 128
        self.drop_path = 0.1 

        # opertimzer settings
        self.clip_grad = None
        self.weight_decay = 0.05
        self.adamW2 = 0.95

        # learning rate settings
        self.lr = None
        self.blr = 1.5e-4
        self.layer_decay = 0.75
        self.min_lr = 1e-6
        self.warmup_epochs = 40

        # augmentation settings
        self.color_jitter = None
        self.aa = 'rand-m9-mstd0.5-inc1'
        self.smoothing = 0.1
        self.reprob = 0.25
        self.remode = 'pixel'
        self.recount = 1
        self.resplit = False

        # mixup settings
        self.mixup = 0.8
        self.cutmix = 1.0
        self.cutmix_minmax = None
        self.mixup_prob = 1.0
        self.mixup_switch_prob = 0.5
        self.mixup_mode = 'batch'

        # loss settings
        self.loss = 'ce'
        self.bal_tau = 1.0
        
        self.mask_ratio = 0.75
        self.global_pool = True # wheather adopt cls token
        self.attn_only = False # https://github.com/facebookresearch/deit/blob/main/README_3things.md

        self.seed = 0
        self.prit = 20 # frequency to print info to console

        self.imbf = 100 # for cifar dataset
        self.num_workers = 16
        self.master_port = 29500


    def get_data_path(self):
        if self.dataset == 'ImageNet-LT':
            self.nb_classes = 1000
            return IMAGENET_LT_PATH
        if self.dataset == 'iNat18':
            self.nb_classes = 8142
            return INAT18_PATH
        if self.dataset == 'ImageNet-BAL':
            self.nb_classes = 1000
            return IMAGENET_BAL_PATH
        if self.dataset == 'cifar10-LT':
            self.nb_classes = 10
            return CIFAR10_PATH
        if self.dataset == 'cifar100-LT':
            self.nb_classes = 100
            return CIFAR100_PATH
        if self.dataset == 'Place':
            self.nb_classes = 365
            return PLCAE_PATH
        return None


    def pretrain(self):
        assert not (self.task == '' or self.note == ''), 'Need basic setting ...'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        task = self.task
        note = self.note
        nodes = len(self.device.split(','))
        data_path = self.get_data_path()
        log_dir = os.path.join(WORK_PATH, f'exp/{task}/{self.dataset}/{note}')
        ckpt_dir = os.path.join(WORK_PATH, f'ckpt/{task}/{self.dataset}/{note}')
        exe_file = os.path.join(WORK_PATH, 'main_pretrain.py')
        os.system(f'''python -m torch.distributed.launch \
                    --nproc_per_node={nodes} \
                    --master_port {self.master_port} \
                    {exe_file} \
                    --ckpt_dir '{ckpt_dir}' \
                    --log_dir '{log_dir}' \
                    --batch_size {self.batch} \
                    --input_size {self.input_size} \
                    --world_size {nodes} \
                    --accum_iter {self.accum_iter} \
                    --model {self.model} \
                    --resume '{self.resume}' \
                    --norm_pix_loss \
                    --mask_ratio {self.mask_ratio} \
                    --epochs {self.epochs} \
                    --warmup_epochs {self.warmup_epochs} \
                    --blr {self.blr} \
                    --weight_decay {self.weight_decay} \
                    --data_path '{data_path}' \
                    --dataset '{self.dataset}' \
                    --num_workers {self.num_workers}
                ''')


    def finetune(self):
        assert not (self.task == '' or self.note == ''), 'Need basic setting ...'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        nodes = len(self.device.split(','))
        data_path = self.get_data_path()
        log_dir = os.path.join(WORK_PATH, f'exp/{self.task}/{self.dataset}/{self.model}/{self.note}')
        ckpt_dir = os.path.join(WORK_PATH, f'ckpt/{self.task}/{self.dataset}/{self.model}/{self.note}')
        exe_file = os.path.join(WORK_PATH, 'main_finetune.py')
        attn_only = '--attn_only' if self.attn_only else ''
        cls_type = '--global_pool' if self.global_pool else '--cls_token'

        os.system(f'''python -m torch.distributed.launch \
                    --nproc_per_node={nodes} \
                    --master_port {self.master_port} \
                    {exe_file}\
                    --ckpt_dir '{ckpt_dir}' \
                    --log_dir '{log_dir}' \
                    --finetune '{self.ckpt}' \
                    --resume '{self.resume}' \
                    --batch_size {self.batch} \
                    --input_size {self.input_size} \
                    --world_size {nodes} \
                    --model {self.model} \
                    --loss {self.loss} \
                    --bal_tau {self.bal_tau} \
                    --accum_iter {self.accum_iter} \
                    --epochs {self.epochs} \
                    --warmup_epochs {self.warmup_epochs} \
                    --blr {self.blr} \
                    --layer_decay {self.layer_decay} \
                    --weight_decay {self.weight_decay} \
                    --adamW2 {self.adamW2} \
                    --drop_path {self.drop_path} \
                    --reprob {self.reprob} \
                    --mixup {self.mixup} \
                    --cutmix {self.cutmix} \
                    --data_path {data_path} \
                    --dataset {self.dataset} \
                    --imbf {self.imbf} \
                    --nb_classes {self.nb_classes} \
                    --num_workers {self.num_workers} \
                    --prit {self.prit} \
                    {attn_only} {cls_type} \
                    --dist_eval
                ''')


    def evaluate(self):
        if self.device != 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        task = self.task
        note = self.note
        nodes = len(self.device.split(','))
        data_path = self.get_data_path()
        log_dir = os.path.join(WORK_PATH, f'exp/{task}/{self.dataset}/{note}')
        exe_file = os.path.join(WORK_PATH, 'main_finetune.py')
        attn_only = '--attn_only' if self.attn_only else ''
        cls_type = '--global_pool' if self.global_pool else '--cls_token'

        os.system(f'''python -m torch.distributed.launch \
                    --nproc_per_node={nodes} \
                    --master_port {self.master_port} \
                    {exe_file}\
                    --log_dir '{log_dir}' \
                    --resume '{self.resume}' \
                    --finetune '{self.finetune}' \
                    --batch_size {self.batch} \
                    --input_size {self.input_size} \
                    --world_size {nodes} \
                    --model {self.model} \
                    --drop_path {self.drop_path} \
                    --data_path {data_path} \
                    --dataset {self.dataset} \
                    --nb_classes {self.nb_classes} \
                    --num_workers {self.num_workers} \
                    --prit {self.prit} \
                    {attn_only} {cls_type} \
                    --eval
                ''')