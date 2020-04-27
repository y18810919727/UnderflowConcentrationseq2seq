#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn


class ASLinear(nn.Linear):

    def __init__(self, features, bias=True, gamma=0.01):
        super(ASLinear, self).__init__(features, features, bias) #set self.weight and self.bias
        self.register_buffer('diff', gamma * torch.eye(features)) #diffusion to avoid bad conditioning

    def forward(self, x):
        return nn.functional.linear(x, self.weight - self.weight.t() - self.diff, self.bias)

class ASRNN(nn.Module):
    """Chang, B., Chi, E. H., Chen, M., & Haber, E. (2019). AntisymmetricRNN:
    A dynamical system view on recurrent neural networks. 7th International
    Conference on Learning Representations, ICLR 2019, (2016), 1–15."""

    """
    This class just implement a cell function for n_seq = 1
    """

    def __init__(self, input_size, num_layers, hidden_size):
        super(ASRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Lx = nn.Linear(input_size, 2*hidden_size, bias=True)
        self.Lh = ASLinear(hidden_size, bias=False, gamma=0.01)

    def forward(self, x, hn=None):

        L, bs, input_size = x.shape
        assert L==1
        assert input_size == self.input_size
        if hn is None:
            hn = torch.zeros(1, bs, self.hidden_size, dtype=x.dtype, device=x.device)

        x = x.squeeze(dim=0)
        hn = hn.squeeze(dim=0)

        Lx = self.Lx(x)
        Lh_gate = self.Lh(hn)
        Lx_gate, Lx_lin = torch.split(Lx, [self.hidden_size, self.hidden_size], dim=1)
        gates = torch.sigmoid(Lx_gate+Lh_gate)
        nh = gates * torch.tanh(Lx_lin + Lh_gate)
        nh = nh.unsqueeze(dim=0)
        return None, nh







    
