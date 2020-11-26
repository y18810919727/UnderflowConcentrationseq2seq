#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import torch
from torch import nn


from models.interpolation import Interpolation, EmptyInterpolation
from models.cells import *



class MIMO(nn.Module):
    def __init__(self, k_in, k_out, k_state, solver='dopri5', stationary=True , ut=0.1, interpolation='cubic',
                 encoder_net_type='GRU', encoder_layers=1, net_type='GRU', adjoint=True, rtol=1e-4, atol=1e-5):
        """

        :param k_in: size of external input
        :param k_out: size of system output
        :param k_state: size of hidden state
        :param solver: optional string indication the integration method to use
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        :param stationary: True or False
        :param ut: Define the time separation in stationary mode
        :param interpolation: 'cubic' only
        :param encoder_net_type: The type of sequential encoder network, RNN or LSTM or GRU.
        :param net_type: the type of differential equation network, RNN or ASRNN or GRU or MLP
        """
        super().__init__()
        self.interpolation = interpolation
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        encoder_net_class_dict = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU,
            'zero_st': ZeroHidden,
            'learn_st': InitialHidden
        }
        self.rnn_encoder = encoder_net_class_dict[encoder_net_type](input_size=k_in + k_out, hidden_size=k_state, num_layers=encoder_layers)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(k_state, 2*k_state, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(2*k_state, k_out, bias=False),
        )
        from models.cells import BasicCell
        self.adjoint = adjoint
        cell = BasicCell(k_state, net_type)
        self.expand_input = nn.Sequential(nn.Linear(k_in, k_state), nn.Tanh())

        class OdeSystem(nn.Module):
            def __init__(self, cell, expand_input, stationary=False, ut=0.1):
                """
                :param cell:
                :param expand_input:
                :param stationary:
                :param ut: Be only useful in stationary mode
                """
                super().__init__()
                self.cell = cell
                self.input_interpolation = EmptyInterpolation()
                self.expand_input = expand_input
                self.stationary = stationary
                self.ut = ut

            def forward(self, t, y):
                diff = self.cell(
                    self.expand_input(self.input_interpolation(t)),
                    y
                )
                if self.stationary:
                    return (diff - y)/self.ut
                else:
                    return diff

            @property
            def input_interpolation(self):
                return self.interpolation

            @input_interpolation.setter
            def input_interpolation(self, interp):
                if not isinstance(interp, Interpolation):
                    raise AttributeError('Unrecognized interpolation module')
                self.input_interpolation = interp

        self.ode_net = OdeSystem(cell, self.expand_input, stationary=stationary, ut=ut)

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

    def forward(self, input, dt=0.1):
        pre_x, pre_y, forward_x = input

        if pre_x.shape[0] == 0:
            pre_x = torch.zeros((1, pre_x.shape[1], pre_x.shape[2])).to(forward_x.device)
            pre_y = torch.zeros((1, pre_y.shape[1], pre_y.shape[2])).to(forward_x.device)
        _, hn = self.rnn_encoder(torch.cat([pre_x, pre_y], dim=2)) # hn (1, batch_size, hidden_num)
        t = torch.linspace(0, dt * forward_x.shape[0], forward_x.shape[0]+1).to(pre_x.device)
        interpolation = Interpolation(
            t,
            torch.cat([pre_x[-1:], forward_x], dim=0),
            method=self.interpolation
        )
        self.ode_net.input_interpolation = interpolation

        if self.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint
        self.ode_net.cell.call_times = 0
        forward_hn_all = odeint(self.ode_net, hn[0], t, rtol=self.rtol, atol=self.atol, method=self.solver)[1:]
        #forward_hn_all = odeint(self.ode_net, hn[0], t, rtol=self.rtol*len(t)/60, atol=self.atol*len(t)/60, method=self.solver)[1:]

        ###############################################################
        # start_hn = hn[0]
        # forward_hn_list = []
        # import time
        # ti = time.time()
        # for i in range(0, len(t)-1, 100):
        #     #cur_t = t[i:min(i+30+1, len(t))]
        #     cur_t_len = min(101, len(t)-i)
        #     cur_t = torch.linspace(0, dt * (cur_t_len-1), cur_t_len)
        #     with torch.no_grad():
        #         forward_hn = odeint(self.ode_net, start_hn, cur_t, rtol=self.rtol, atol=self.atol, method=self.solver)[1:]
        #     forward_hn_list.append(forward_hn)
        #     start_hn = forward_hn[-1]
        #     print(cur_t)
        #     print('{}-{}-{}s-{}-mean:{}-var:{}'.format(
        #         len(t), i, time.time()-ti, self.ode_net.cell.call_times, torch.mean(forward_hn), torch.var(forward_hn))
        #     )
        #     ti = time.time()
        # forward_hn_all = torch.cat(forward_hn_list, dim=0)
        ###############################################################
        estimate_y_all = self.recursive_predict(forward_hn_all, max_len=1000)
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







