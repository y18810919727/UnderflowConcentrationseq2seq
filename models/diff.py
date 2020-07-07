#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

from torch import nn
import torch
from torch.nn import RNN, Module, LSTM, GRU

class DiffNet(Module):
    def __init__(self, input_size, num_layers, hidden_size, out_size, config, net_type='lstm'):
        """

        :param input_size: x's shape
        :param hidden_size:
        :param outsize: y's shape
        """
        super(DiffNet, self).__init__()
        self.config = config

        self.net_type = net_type
        self.rnn = self.rnn_init(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size)

        self.fc = self.fc_init(hidden_size=hidden_size, out_size=out_size)

        self.parameters_init()

    def rnn_init(self, input_size, num_layers, hidden_size):

        if self.config.begin == 'rnn_st':
            rnn = RNN(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size)
        elif self.config.begin == 'lstm_st':

            class SingleLSTM(nn.Module):

                def __init__(self, _input_size, _num_layers, _hidden_size):
                    super(SingleLSTM, self).__init__()
                    self.lstm = LSTM(input_size=_input_size, num_layers=_num_layers, hidden_size=_hidden_size)

                def forward(self, input, hx=None):
                    output, hn = self.lstm(input, hx)
                    return output, hn[0]
            rnn = SingleLSTM(_input_size=input_size, _num_layers=num_layers, _hidden_size=hidden_size)

        elif self.config.begin == 'GRU_st':
            rnn = GRU(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size)

        elif self.config.begin in ['learn_st', 'zero_st']:
            class InitialHidden(nn.Module):

                def __init__(self, hidden_size, keep_zero):
                    super(InitialHidden, self).__init__()
                    self.hidden_size = hidden_size
                    self.begin_state = torch.nn.Parameter(torch.randn(hidden_size))
                    self.keep_zero = keep_zero
                def forward(self, x):
                    _, bs, _=x.shape
                    hn = self.begin_state.repeat(bs, 1).unsqueeze(dim=0).to(x.device)
                    if self.keep_zero:
                        hn = hn *0
                    output = hn
                    return output, hn
            rnn = InitialHidden(hidden_size, self.config.begin == 'zero_st')
        else:
            raise AttributeError()

        # net_type = self.net_type
        #
        # if net_type == 'rnn':
        # elif net_type == 'lstm':
        #     rnn = LSTM(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size)
        # elif net_type == 'GRU':
        #     rnn = GRU(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size)
        # else:
        #     raise ValueError("can not identify net_type.")
        return rnn

    def fc_init(self, hidden_size, out_size):

        fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, out_size, bias=False),
        )
        return fc

    def parameters_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)

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



