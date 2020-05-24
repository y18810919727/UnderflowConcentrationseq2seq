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
from custom_dataset import Target_Col, Control_Col

class ODENet(nn.Module):
    def __init__(self, grad_module, t_step, interpolation_kind):
        super(ODENet, self).__init__()
        self.grad_module = grad_module
        self.t_step = t_step
        self.u_seq = None
        self.cum_t = 0
        self.fit_fcn_list = None
        self.interpolation_kind = interpolation_kind

    def set_u(self, u_seq=None):
        self.u_seq = u_seq

    def forward(self, t, y):

        if self.interpolation_kind == 'linear':

            t_range = self.t_step * (len(self.u_seq) - 1)
            # When ode solver calculate y(t), it needs the gradient of y to (t+dt),so the parameter t may exceeds the t_range.
            t = torch.clamp(t, 0, t_range)
            self.cum_t += 1
            u_position = float((self.u_seq.shape[0] - 1)*t.cpu()/t_range)

            u_index = int(u_position)
            u_index_plus_one = min(u_index + 1, self.u_seq.shape[0]-1)

            u_left = self.u_seq[u_index]
            u_right = self.u_seq[u_index_plus_one]
            cur_u = u_left + (u_right - u_left) * (u_position - u_index)
        else:
            assert len(self.fit_fcn_list) == y.shape[0]
            cur_u = torch.stack([f(t) for f in self.fit_fcn_list], dim=0)

        #cur_u = cur_u * 0
        cur_u = cur_u.to(y.device)
        return self.grad_module(
            torch.cat([y, cur_u], dim=1)
        )


    def fit_c_u_seq(self, c_u_seq, ode_t, seq_t=None):
        if seq_t is None:
            seq_t = torch.arange(c_u_seq.shape[0]).to(c_u_seq)*self.t_step

        beg_int_index = int(ode_t[0]/self.t_step)
        end_int_index = min(int(ode_t[-1]/self.t_step-1e-9) + 1, c_u_seq.shape[0]-1)
        x = seq_t[beg_int_index: end_int_index + 1]
        batch_size = c_u_seq.shape[1]
        self.fit_fcn_list = []

        class Interpolation1D2nD:

            def __init__(self, n, X, Y, kind):
                X = X.cpu().numpy()
                Y = Y.cpu().numpy()
                from scipy.interpolate import interp1d
                assert len(X.shape) == 1 and len(Y.shape) == 2
                self.n = n
                assert n == Y.shape[1]

                self.inter_f = [interp1d(X, Y[:, i], kind=kind, fill_value='extrapolate') for i in range(n)]

            def __call__(self, X):
                device = X.device
                X = X.cpu().detach().numpy()
                res = np.stack([f(X) for f in self.inter_f], axis=0)
                return torch.FloatTensor(res).to(device)

        for i in range(batch_size):
            y = c_u_seq[beg_int_index: end_int_index + 1, i, :]
            self.fit_fcn_list.append(Interpolation1D2nD(y.shape[1], x, y, self.interpolation_kind))





class MyODE(DiffNet):

    def __init__(self, input_size, num_layers, hidden_size, out_size, config, net_type='lstm'):

        super(MyODE, self).__init__(input_size, num_layers, hidden_size, out_size, config, net_type)

        modules_list = []
        hidden_sizes = num_layers * [hidden_size]
        layer_sizes = [hidden_size+len(Control_Col)] + hidden_sizes
        self.net_type = net_type


        for i in range(1, len(layer_sizes)):
            modules_list.append(
                nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            )
            modules_list.append(nn.Tanh())
        modules_list.append(
            nn.Linear(layer_sizes[-1], hidden_size)
        )

        self.ode_net = ODENet(
            nn.Sequential(*modules_list),
            self.config.t_step,
            config.interpolation,
        )



    def forward(self, input, only_forward_y=True):
        """

        :param input: pre_x, pre_y, forward_x = input
        shape (seq_len, batch, input_size):
        :return:
        """
        pre_x, pre_y, forward_x = input
        output, hn = self.rnn(torch.cat([pre_x, pre_y], dim=2))

        t = torch.linspace(0, self.config.t_step * forward_x.shape[0], forward_x.shape[0]+1)
        if self.config.use_cuda:
            t = t.cuda()
        self.ode_net.set_u(torch.cat([pre_x[-1:], forward_x], dim=0))
        self.ode_net.cum_t = 0
        from common import discrete_odeint
        hn_all = discrete_odeint(self.ode_net, torch.cat([pre_x[-1:], forward_x], dim=0),
                                 hn[0], t, rtol=self.config.rtol, atol=self.config.atol, method=self.config.ode_method)[1:]
        #print('forward %i %f s' % (self.ode_net.cum_t, time.time() - time_beg))

        estimate_y_all = self.recursive_predict(hn_all, max_len=50)
        return estimate_y_all




if __name__ == '__main__':
    print(111)