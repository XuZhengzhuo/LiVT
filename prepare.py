import numpy as np
import os
from tqdm import tqdm
import json
import shutil
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def create_iNat18(task='train'):
    src_root = '/path/to/iNat18'
    dst_root = '/diskC/xzz/iNat18'

    if not os.path.exists(os.path.join(dst_root, task, '0')):
        for i in range(8142):
            pth = os.path.join(dst_root, task, str(i))
            os.makedirs(pth, exist_ok=True)

    with open(os.path.join(src_root, f'iNat18_{task}.json')) as f:
        meta = json.load(f)
        for anno in tqdm(meta['annotations']):
            img_path = anno['fpath'].replace('./downloaed/iNat18/','')
            img_name = anno['fpath'].split('/')[-1]
            cls_name = str(anno['category_id'])
            src_pth = os.path.join(src_root, img_path)
            dst_pth = os.path.join(dst_root, task, cls_name, img_name)
            shutil.copy(src_pth, dst_pth)


def create_ImageNet_BAL(task='train'):
    src_root = f'/path/to/ImageNet/{task}'
    dst_root = f'/diskC/xzz/ImageNet_BAL/{task}'
    img_cnt = 160
    
    folds = os.listdir(src_root)
    for fold in tqdm(folds):
        fold_src = os.path.join(src_root, fold)
        fold_dst = os.path.join(dst_root, fold)
        os.makedirs(fold_dst, exist_ok=True)
        imgs = os.listdir(fold_src)
        for img in imgs[:img_cnt]:
            src_pth = os.path.join(fold_src, img)
            dst_pth = os.path.join(fold_dst, img)
            shutil.copy(src_pth, dst_pth)
