#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

import torch
from torch import nn


class ASLinear(nn.Linear):

    def __init__(self, features, bias=True, gamma=0.01):
        super(ASLinear, self).__init__(features, features, bias) #set self.weight and self.bias
        self.register_buffer('diff', gamma * torch.eye(features)) #diffusion to avoid bad conditioning

    def forward(self, x):
        return nn.functional.linear(x, self.weight - self.weight.t() - self.diff, self.bias)

class ASRNNCell(nn.Module):
    """Chang, B., Chi, E. H., Chen, M., & Haber, E. (2019). AntisymmetricRNN:
    A dynamical system view on recurrent neural networks. 7th International
    Conference on Learning Representations, ICLR 2019, (2016), 1–15."""

    """
    This class just implement a cell function for n_seq = 1
    """

    def __init__(self,  input_size, hidden_size):
        super(ASRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Lh = ASLinear(hidden_size, bias=False, gamma=0.01)
        self.Lx = nn.Linear(input_size, 2 * hidden_size, bias=True)

    def forward(self, x, hn=None):

        bs, input_size = x.shape
        if hn is None:
            hn = torch.zeros(bs, self.hidden_size, dtype=x.dtype, device=x.device)
        Lh_gate = self.Lh(hn)
        Lx = self.Lx(x)
        Lx_gate, Lx_lin = torch.split(Lx, [self.hidden_size, self.hidden_size], dim=1)
        gates = torch.sigmoid(Lx_gate+Lh_gate)
        nh = gates * torch.tanh(Lx_lin + Lh_gate)
        return nh

class MLPCell(nn.Module):

    def __init__(self,  input_size, hidden_size):
        super(MLPCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Lh = ASLinear(hidden_size, bias=False, gamma=0.01)
        self.Lx = nn.Linear(input_size, 2 * hidden_size, bias=True)

        layer_sizes = [input_size+hidden_size, 3 * hidden_size]

        modules_list = []
        for i in range(1, len(layer_sizes)):
            modules_list.append(
                nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            )
            modules_list.append(nn.Tanh())
        modules_list.append(
            nn.Linear(layer_sizes[-1], hidden_size)
        )
        self.mlp = nn.Sequential(*modules_list)


    def forward(self, x, hn):
        return self.mlp(torch.cat([x, hn], dim=1))


class BasicCell(nn.Module):
    def __init__(self, k_state, net_type='GRU'):
        super().__init__()
        self.cell = None
        self.state0 = nn.Parameter(torch.zeros(k_state,), requires_grad=True)
        cell_factory_dict = {
            'GRU': nn.GRUCell,
            'RNN': nn.RNNCell,
            'ASRNN': ASRNNCell,
            'MLP': MLPCell
        }

        if net_type not in cell_factory_dict.keys():
            raise AttributeError('Cell type {} is not implemented'.format(net_type))

        self.cell = cell_factory_dict[net_type](k_state, k_state)
        self.call_times = 0

    def forward(self, x, hn=None):
        if hn is None:
            hn = self.state0.unsqueeze(0).expand(x.shape[1], -1)

        self.call_times += 1
        return self.cell(x, hn)





