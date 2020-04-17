#!/usr/bin/python
# -*- coding:utf8 -*-
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
            SummaryWriter.add_figure(self)
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
