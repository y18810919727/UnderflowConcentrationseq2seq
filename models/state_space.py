#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
import torch.nn.functional as F


class StateSpace(torch.nn.Module):
    def __init__(self, k_in, k_out, k_state, max_length=80):

        """
        :param k_in: size of external input
        :param k_out: size of system output
        :param k_state: size of hidden state
        :param encoder_net_type: The type of sequential encoder network, RNN or LSTM or GRU.
        """
        super(StateSpace, self).__init__()
        self.max_length = max_length
        self.A = nn.Linear(k_state, k_state)
        self.B = nn.Linear(k_in, k_state)
        self.C = nn.Linear(k_state, k_out)
        self.rnn_encoder = nn.RNN(k_in + k_out, k_state, num_layers=1)

    def forward(self, input):

        pre_x, pre_y, forward_x = input

        if pre_x.shape[0] == 0:
            pre_x = torch.zeros((1, pre_x.shape[1], pre_x.shape[2])).to(forward_x.device)
            pre_y = torch.zeros((1, pre_y.shape[1], pre_y.shape[2])).to(forward_x.device)

        _, hn = self.rnn_encoder(torch.cat([pre_x, pre_y], dim=2))  # hn (1, batch_size, hidden_num)
        hn = hn[0]
        forward_length, batch_size, _ = forward_x.shape
        # import pdb
        # pdb.set_trace()
        assert pre_y.size()[0] <= self.max_length

        estimate_y_list = []
        for di in range(forward_length):
            hn = self.A(hn) + self.B(forward_x[di])
            estimate_y_list.append(self.C(hn))
        estimate_y_all = torch.stack(estimate_y_list, dim=0)

        return estimate_y_all
