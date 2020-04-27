#!/usr/bin/python
# -*- coding:utf8 -*-
import re
import numpy as np
import math
import os
import json

import torch

from tensorboardX import SummaryWriter

def cal_params_sum(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

class MyWriter(SummaryWriter):
    def __init__(self, save_path, is_write):
        self.is_write = is_write
        if is_write:
            super(MyWriter, self).__init__(save_path)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if self.is_write:
            SummaryWriter.add_scalar(self, tag, scalar_value, global_step, walltime)

    def add_pr_curve(self, tag, labels, predictions, global_step=None,
                     num_thresholds=127, weights=None, walltime=None):

        if self.is_write:
            SummaryWriter.add_pr_curve(self, tag, labels, predictions, global_step,
                                       num_thresholds, weights, walltime)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):

        if self.is_write:
            SummaryWriter.add_figure(self, tag, figure, global_step, close, walltime)

    def close(self):

        if self.is_write:
            SummaryWriter.close(self)
        else:
            pass



def make_fcn(input_size, num_layers, hidden_size, out_size):

    from torch import nn
    modules_list = []
    layer_sizes = [input_size] + num_layers * [hidden_size]

    for i in range(1, len(layer_sizes)):
        modules_list.append(
            nn.Linear(layer_sizes[i - 1], layer_sizes[i])
        )
        modules_list.append(nn.Tanh())
    modules_list.append(
        nn.Linear(layer_sizes[-1], out_size)
    )
    return nn.Sequential(*modules_list)


def col2Index(all_col, col):
    return [list(all_col).index(x) for x in col]

def RRSE(y_pred, y_gt):
    assert y_gt.shape == y_pred.shape
    if len(y_gt.shape) == 3:
        return torch.mean(
            torch.stack(
                [RRSE(y_pred[:, i], y_gt[:, i]) for i in range(y_gt.shape[1])]
            )
        )

    elif len(y_gt.shape) == 2:
        # each shape (n_seq, n_outputs)
        se = torch.sum((y_gt - y_pred)**2, dim=0)
        rse = se / torch.sum(
            (y_gt - torch.mean(y_gt, dim=0))**2, dim=0
        )
        return torch.mean(torch.sqrt(rse))
    else:
        raise AttributeError


class GaussLoss(torch.nn.Module):
    def __init__(self, n, config):
        super(GaussLoss, self).__init__()
        self.n = n
        self.sigma = torch.randn(n)
        if config.use_cuda:
            self.sigma = self.sigma.cuda()

        self.sigma = torch.nn.Parameter(self.sigma)


    def get_cov(self):
        sigma = torch.nn.functional.softplus(self.sigma)
        return torch.diag(sigma)

    def __call__(self, pred, gt):
        from torch.distributions.multivariate_normal import MultivariateNormal
        mu = MultivariateNormal(pred, self.get_cov())
        return -torch.sum(mu.log_prob(gt))


def parser_dir(dir_name, config):
    net_type = dir_name[:dir_name.find('_')]
    config.net_type = net_type

    dis = re.findall('_dis(\d*)', dir_name)
    if len(dis)>0:
        config.sample_dis = int(dis[0])

    hidden = re.findall('_h(\d{2})', dir_name)
    if len(hidden) >=1:
        config.hidden_num = int(hidden[0])

    if 'ode' in dir_name:
        if 'affine' in dir_name:
            config.algorithm = 'ode_affine'
        else:
            config.algorithm = 'ode'

        tols = re.findall('_(\d)_(\d)', dir_name)
        config.rtol = float('1e-' + tols[0][0])
        config.atol = float('1e-' + tols[0][1])
    elif 'RK' in dir_name:
        RK_order = re.findall('RK(\d)', dir_name)[0]
        config.algorithm = 'RK' + RK_order

    elif '_diff' in dir_name:
        config.algorithm = 'hidden_rnn'
        config.no_hidden_diff = False

    elif '_nodiff' in dir_name:
        config.algorithm = 'hidden_rnn'
        config.no_hidden_diff = True
    else:
        return False

    return True

if __name__ == '__main__':

    y = torch.rand((3,3,3))*100
    x = torch.rand((3,3,3)) + y
    print(RRSE(x,y))