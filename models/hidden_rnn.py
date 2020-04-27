#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from models.diff import DiffNet
from models.ode import ODENet
from custom_dataset import Target_Col, Control_Col
from config import args as config
from models.asrnn import ASRNN

from torch import nn

class HiddenRNN(DiffNet):
    def __init__(self, input_size, num_layers, hidden_size, out_size, net_type='lstm', epsilon=0.1):

        super(HiddenRNN, self).__init__(input_size, num_layers, hidden_size, out_size, net_type)

        self.epsilon = 0.1
        if config.net_type == 'rnn':
            Model_Class = nn.RNN
        elif config.net_type == 'lstm':
            Model_Class = nn.LSTM
        elif config.net_type == 'GRU':
            Model_Class = nn.GRU
        elif config.net_type == 'asrnn':
            Model_Class =ASRNN
        else:
            raise AttributeError("Please give a type of rnn net. lstm or rnn or GRU or asrnn")

        self.hidden_rnn = Model_Class(input_size=len(Control_Col), num_layers=num_layers, hidden_size=hidden_size)


    def forward(self, input, only_forward_y=True):
        """

        :param input: pre_x, pre_y, forward_x = input
        shape (seq_len, batch, input_size):
        :return:
        """
        pre_x, pre_y, forward_x = input
        output, hn = self.rnn(torch.cat([pre_x, pre_y], dim=2))


        estimate_y_list = []
        for i in range(forward_x.shape[0]):
            estimate_y = self.fc(output[-1])
            estimate_y_list.append(estimate_y)

            # When n_seq = 1, hn = output, so the output of RNN is useless.
            _, hn_inc = self.hidden_rnn(torch.unsqueeze(
                forward_x[i], dim=0
            ), hn)
            if config.net_type == 'rnn' or config.net_type == 'GRU' or config.net_type == 'asrnn':
                if config.no_hidden_diff:
                    hn = hn_inc
                else:
                    hn = hn + self.epsilon * hn_inc
                output = hn
            else:
                if config.no_hidden_diff:
                    hn = hn_inc
                else:
                    hn = (hn[0] + self.epsilon * hn_inc[0], hn_inc[1])
                output = hn[0]

        estimate_y_all = torch.stack(estimate_y_list, dim=0)
        return estimate_y_all


