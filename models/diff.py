#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

from torch import nn
import torch
from torch.nn import RNN, Module, LSTM

class DiffNet(Module):
    def __init__(self, input_size, num_layers, hidden_size, out_size, net_type='lstm'):
        """

        :param input_size: x's shape
        :param hidden_size:
        :param outsize: y's shape
        """
        super(DiffNet, self).__init__()
        self.rnn = self.rnn_init(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size, net_type=net_type)

        self.fc = self.fc_init(hidden_size=hidden_size, out_size=out_size)

        self.parameters_init()

    def rnn_init(self, input_size, num_layers, hidden_size, net_type='lstm'):
        if net_type == 'rnn':
            rnn = RNN(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size)
        elif net_type == 'lstm':
            rnn = LSTM(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size)
        else:
            raise ValueError("can not identify net_type.")
        return rnn

    def fc_init(self, hidden_size, out_size):

        fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, out_size)
        )
        return fc

    def parameters_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)

    def forward(self, input):
        """

        :param input: pre_x, pre_y, forward_x = input
        shape (seq_len, batch, input_size):
        :return:
        """
        pre_x, pre_y, forward_x = input
        output, hn = self.rnn(torch.cat([pre_x, pre_y], dim=2))
        estimate_y_list = []
        last_y = pre_y[-1]
        for i in range(forward_x.shape[0]):
            estimate_y = self.fc(output[-1]) + last_y
            estimate_y_list.append(estimate_y)

            output, hn = self.rnn(torch.unsqueeze(
                torch.cat([forward_x[i], estimate_y], dim=1), dim=0
            ), hn)
            last_y = estimate_y

        estimate_y_all = torch.stack(estimate_y_list, dim=0)

        return estimate_y_all

    def recursive_predict(self, hn_seq, max_len=50):

        len, bs, hidden_size = hn_seq.shape
        if hn_seq.shape[0]<=max_len:
            return self.fc(
                hn_seq.view(-1, hidden_size)
            ).view(len, bs, -1)
        return torch.cat(
            [
                self.recursive_predict(hn_seq[0:len//2], max_len),
                self.recursive_predict(hn_seq[len//2:], max_len)
            ], dim=0
        )



