#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
import time

import torch
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint
from torch.nn import RNN
from models.diff import DiffNet
from torch import nn
from config import args as config
from custom_dataset import Target_Col, Control_Col

class ODENet(nn.Module):
    def __init__(self, grad_module):
        super(ODENet, self).__init__()
        self.grad_module = grad_module
        self.u_seq = None
        self.cum_t = 0

    def set_u(self, u_seq=None):
        self.u_seq = u_seq

    def forward(self, t, y):
        t_range = config.t_step * (len(self.u_seq) - 1)

        # When ode solver calculate y(t), it needs the gradient of y to (t+dt),so the parameter t may exceeds the t_range.
        t = torch.clamp(t, 0, t_range)
        self.cum_t += 1
        u_position = float((self.u_seq.shape[0] - 1)*t.cpu()/t_range)

        u_index = int(u_position)
        u_index_plus_one = min(u_index + 1, self.u_seq.shape[0]-1)

        u_left = self.u_seq[u_index]
        u_right = self.u_seq[u_index_plus_one]
        cur_u = u_left + (u_right - u_left) * (u_position - u_index)
        #cur_u = cur_u * 0
        if config.use_cuda:
            cur_u = cur_u.cuda()
        return self.grad_module(
            torch.cat([y, cur_u], dim=1)
        )




class MyODE(DiffNet):

    def __init__(self, input_size, num_layers, hidden_size, out_size, net_type='lstm'):

        super(MyODE, self).__init__(input_size, num_layers, hidden_size, out_size, net_type)

        modules_list = []
        hidden_sizes = num_layers * [hidden_size]
        layer_sizes = [hidden_size+len(Control_Col)] + hidden_sizes


        for i in range(1, len(layer_sizes)):
            modules_list.append(
                nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            )
            modules_list.append(nn.Tanh())
        modules_list.append(
            nn.Linear(layer_sizes[-1], hidden_size)
        )

        self.ode_net = ODENet(
            nn.Sequential(*modules_list)
        )



    def forward(self, input, only_forward_y=True):
        """

        :param input: pre_x, pre_y, forward_x = input
        shape (seq_len, batch, input_size):
        :return:
        """
        pre_x, pre_y, forward_x = input
        output, hn = self.rnn(torch.cat([pre_x, pre_y], dim=2))
        if config.net_type == 'lstm':
            hn = hn[0]

        t = torch.linspace(0, config.t_step * forward_x.shape[0], forward_x.shape[0]+1)
        if config.use_cuda:
            t = t.cuda()
        self.ode_net.set_u(torch.cat([pre_x[-1:], forward_x], dim=0))
        self.ode_net.cum_t = 0
        time_beg = time.time()
        hn_all = odeint(self.ode_net, hn[0], t, rtol=config.rtol, atol=config.atol)[1:]
        #print('forward %i %f s' % (self.ode_net.cum_t, time.time() - time_beg))

        estimate_y_all = self.recursive_predict(hn_all, max_len=50)
        return estimate_y_all




if __name__ == '__main__':
    print(111)