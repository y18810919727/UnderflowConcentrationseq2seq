#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from .ode import MyODE, ODENet
from torchdiffeq import odeint as odeint
import time
from torch import nn
from common import make_fcn




class MyODEAffine(MyODE):

    def __init__(self, input_size, num_layers, hidden_size, out_size, config, net_type='lstm'):

        super(MyODE, self).__init__(input_size, num_layers, hidden_size, out_size, config, net_type)

        self.f = make_fcn(hidden_size, 1, 16, hidden_size)
        self.g = make_fcn(hidden_size, 1, 16, hidden_size * len(config.Control_Col))
        self.Control_Col = config.Control_Col

        class AffineGradientMoudle(nn.Module):
            def __init__(self, f, g, x_size, u_size):
                super(AffineGradientMoudle, self).__init__()
                self.f = f
                self.g = g
                self.x_size = x_size
                self.u_size = u_size

            def forward(self, input):
                """

                :param input:
                :return: \gradient(x) = f(x) + g(x)*u
                """
                x = input[:,:self.x_size]
                u = input[:, self.x_size:]

                return self.f(x) + torch.matmul(self.g(x).contiguous().view(-1, self.x_size, self.u_size),
                                                u.contiguous().view(-1, self.u_size, 1)).squeeze(-1)

        self.ode_net = ODENet(AffineGradientMoudle(self.f, self.g, hidden_size, len(self.Control_Col)), self.config.t_step,
                              self.config.interpolation)

