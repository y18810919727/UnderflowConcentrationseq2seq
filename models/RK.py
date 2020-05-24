#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

from models.ode import ODENet

from models.ode import MyODE



class RK(MyODE):
    def __init__(self, input_size, num_layers, hidden_size, out_size, config, net_type='rnn'):
        super(RK, self).__init__(input_size, num_layers, hidden_size, out_size, config, net_type)
        self.grad_module = self.ode_net.grad_module
        self.ode_net = None


    def forward(self, input, only_forward_y=True):
        """

        :param input: pre_x, pre_y, forward_x = input
        shape (seq_len, batch, input_size):
        :return:
        """
        pre_x, pre_y, forward_x = input
        output, hn = self.rnn(torch.cat([pre_x, pre_y], dim=2))
        if self.config.net_type == 'lstm':
            hn = hn[0]

        t = torch.linspace(0, self.config.t_step * forward_x.shape[0], forward_x.shape[0]+1)
        if self.config.use_cuda:
            t = t.cuda()


        u_seq = torch.cat([pre_x[-1:], forward_x], dim=0)

        # hn.shape (1,bs,h) -> (bs,h)
        hn = hn[0]

        hn_all_list = []

        for i in range(len(t)-1):
            mid_u = (u_seq[i] + u_seq[i+1])/2
            dt = t[i+1]-t[i]

            hn = hn + self.cal_RK_parameters(u_seq[i], u_seq[i+1], hn, dt, int(self.config.algorithm[-1]))
            hn_all_list.append(hn)

        hn_all = torch.stack(hn_all_list, dim=0)
        #print('forward %i %f s' % (self.ode_net.cum_t, time.time() - time_beg))
        estimate_y_all = self.recursive_predict(hn_all, max_len=50)
        return estimate_y_all

    def cal_RK_parameters(self, x_i, x_i_plus_1, hn , dt, type=4):

        mid_x = (x_i + x_i_plus_1) / 2
        def tc(a,b):
            return torch.cat([a,b], dim=1)

        k0 = self.grad_module(tc(hn, x_i))
        if type == 4:

            k1 = self.grad_module(tc(hn+0.5*dt*k0, mid_x))
            k2 = self.grad_module(tc(hn+0.5*dt*k1, mid_x))
            k3 = self.grad_module(tc(hn+dt*k2, x_i_plus_1))
            b = [1/6, 1/3,1/3,1/6]
            k = [k0, k1, k2, k3]
        elif type == 3:

            k1 = self.grad_module(tc(hn+0.5*dt*k0, mid_x))
            k2 = self.grad_module(tc(hn-dt*k0+2*dt*k1, x_i_plus_1))
            b = [1/6,2/3,1/6]
            k = [k0, k1, k2]
        elif type == 2:
            k1 = self.grad_module(tc(hn+0.5*dt*k0, mid_x))
            b = [1]
            k = [k1]
        elif type == 1:
            b = [1]
            k = [k0]
        else:
            raise AttributeError

        return sum([a*b for a, b in zip(b, k)]) * dt



