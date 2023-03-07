import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy
from typing import Optional


class ST_CE_loss(nn.Module):
    """
        CE loss, timm implementation for mixup
    """
    def __init__(self):
        super(ST_CE_loss, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class Bal_CE_loss(nn.Module):
    '''
        Paper: https://arxiv.org/abs/2007.07314
        Code: https://github.com/google-research/google-research/tree/master/logit_adjustment
    '''
    def __init__(self, args):
        super(Bal_CE_loss, self).__init__()
        prior = np.array(args.cls_num)
        prior = np.log(prior / np.sum(prior))
        prior = torch.from_numpy(prior).type(torch.FloatTensor)
        self.prior = args.bal_tau * prior

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prior = self.prior.to(x.device)
        x = x + prior
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class BCE_loss(nn.Module):

    def __init__(self, args,
                target_threshold=None, 
                type=None,
                reduction='mean', 
                pos_weight=None):
        super(BCE_loss, self).__init__()
        self.lam = 1.
        self.K = 1.
        self.smoothing = args.smoothing
        self.target_threshold = target_threshold
        self.weight = None
        self.pi = None
        self.reduction = reduction
        self.register_buffer('pos_weight', pos_weight)

        if type == 'Bal':
            self._cal_bal_pi(args)
        if type == 'CB':
            self._cal_cb_weight(args)

    def _cal_bal_pi(self, args):
        cls_num = torch.Tensor(args.cls_num)
        self.pi = cls_num / torch.sum(cls_num)

    def _cal_cb_weight(self, args):
        eff_beta = 0.9999
        effective_num = 1.0 - np.power(eff_beta, args.cls_num)
        per_cls_weights = (1.0 - eff_beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(args.cls_num)
        self.weight = torch.FloatTensor(per_cls_weights).to(args.device)

    def _bal_sigmod_bias(self, x):
        pi = self.pi.to(x.device)
        bias = torch.log(pi) - torch.log(1-pi)
        x = x + self.K * bias
        return x

    def _neg_reg(self, labels, logits, weight=None):
        if weight == None:
            weight = torch.ones_like(labels).to(logits.device)
        pi = self.pi.to(logits.device)
        bias = torch.log(pi) - torch.log(1-pi)
        logits = logits * (1 - labels) * self.lam + logits * labels # neg + pos
        logits = logits + self.K * bias
        weight = weight / self.lam * (1 - labels) + weight * labels # neg + pos
        return logits, weight

    def _one_hot(self, x, target):
        num_classes = x.shape[-1]
        off_value = self.smoothing / num_classes
        on_value = 1. - self.smoothing + off_value
        target = target.long().view(-1, 1)
        target = torch.full((target.size()[0], num_classes),
            off_value, device=x.device, 
            dtype=x.dtype).scatter_(1, target, on_value)
        return target

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            target = self._one_hot(x, target)
        if self.target_threshold is not None:
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        weight = self.weight
        if self.pi != None: x = self._bal_sigmod_bias(x)
        # if self.lam != None:
        #     x, weight = self._neg_reg(target, x)
        C = x.shape[-1] # + log C
        return C * F.binary_cross_entropy_with_logits(
                    x, target, weight, self.pos_weight,
                    reduction=self.reduction)


class LS_CE_loss(nn.Module):
    """
        label smoothing without mixup
    """
    def __init__(self, smoothing=0.1):
        super(LS_CE_loss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MiSLAS_loss(nn.Module):
    ''' 
        Paper: Improving Calibration for Long-Tailed Recognition
        Code: https://github.com/Jia-Research-Lab/MiSLAS
    '''
    def __init__(self, args, shape='concave', power=None):
        super(MiSLAS_loss, self).__init__()

        cls_num_list = args.cls_num
        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)
        smooth_head = 0.3
        smooth_tail = 0.0

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)
        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()

    def forward_oneway(self, x, target):
        smooth = self.smooth.to(x.device)
        smoothing = smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        loss = 0
        if target.shape == x.shape: # to match mixup
            '''
                x.shape: batch * nClass
                target: one hot [0, 0, 0, 0.4, 0, 0, 0.6, 0, 0, 0]
            '''
            _, idx_ = torch.topk(target, k=2, dim=1, largest=True)
            i1, i2 = idx_[:,0], idx_[:,1]
            v1 = target[torch.tensor([i for i in range(x.shape[0])]), i1]
            v2 = target[torch.tensor([i for i in range(x.shape[0])]), i2]
            loss_y1 = self.forward_oneway(x, i1)
            loss_y2 = self.forward_oneway(x, i2)
            loss = v1.mul(loss_y1) + v2.mul(loss_y2)
        else:
            loss = self.forward_oneway(x, target)
        return loss.mean()


class LADE_loss(nn.Module):
    '''NOTE can not work with mixup, plz set mixup=0 and cutmix=0
        Paper: Disentangling Label Distribution for Long-tailed Visual Recognition
        Code: https://github.com/hyperconnect/LADE
    '''
    def __init__(self, args, remine_lambda=0.1):
        super().__init__()
        cls_num = torch.tensor(args.cls_num)
        self.prior = cls_num / torch.sum(cls_num)
        self.num_classes = args.nb_classes
        self.balanced_prior = torch.tensor(1. / self.num_classes).float()
        self.remine_lambda = remine_lambda
        self.cls_weight = (cls_num.float() / torch.sum(cls_num.float()))

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)
        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, x, target, q_pred=None):
        """
            x: N x C
            target: N
        """
        prior = self.prior.to(x.device)
        balanced_prior = self.balanced_prior.to(x.device)
        cls_weight = self.cls_weight.to(x.device)
        per_cls_pred_spread = x.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (x - torch.log(prior + 1e-9) + torch.log(balanced_prior + 1e-9)).T  # C x N
        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, _, _ = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)
        return - torch.sum(estim_loss * cls_weight)


class LDAM_loss(nn.Module):
    '''
        Paper: Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
        Code: https://github.com/kaidic/LDAM-DRW
    '''
    def __init__(self, args):
        super(LDAM_loss, self).__init__()
        cls_num_list = args.cls_num
        self.drw = False
        self.epoch = 0
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (0.5 / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        self.s = 30

    def forward_oneway(self, x, target):
        m_list = self.m_list.to(x.device)
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor).to(x.device)
        batch_m = torch.matmul(m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, reduction='none')

    def forward(self, x, target):
        loss = 0
        if target.shape == x.shape: # to match mixup
            _, idx_ = torch.topk(target, k=2, dim=1, largest=True)
            i1, i2 = idx_[:,0], idx_[:,1]
            v1 = target[torch.tensor([i for i in range(x.shape[0])]), i1]
            v2 = target[torch.tensor([i for i in range(x.shape[0])]), i2]
            loss_y1 = self.forward_oneway(x, i1)
            loss_y2 = self.forward_oneway(x, i2)
            loss = v1.mul(loss_y1) + v2.mul(loss_y2)
        else:
            loss = self.forward_oneway(x, target)
        return loss.mean()


class CB_CE_loss(nn.Module):
    '''
        Paper: Class-Balanced Loss Based on Effective Number of Samples
        Code: https://github.com/richardaecn/class-balanced-loss
    '''
    def __init__(self, args):
        super(CB_CE_loss, self).__init__()
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, args.cls_num)
        weight = (1.0 - beta) / np.array(effective_num)
        weight = weight / np.sum(weight) * len(args.cls_num)
        self.weight = torch.FloatTensor(weight)

    def forward(self, x, target):
        weight = self.weight.to(x.device)
        return F.cross_entropy(input = x, target = target, weight = weight)
