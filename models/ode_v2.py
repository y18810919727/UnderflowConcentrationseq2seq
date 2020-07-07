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




class MyODEV2(MyODE):

    def __init__(self, input_size, num_layers, hidden_size, out_size, config, net_type='lstm'):

        super(MyODE, self).__init__(input_size, num_layers, hidden_size, out_size, config, net_type)

        modules_list = []
        hidden_sizes = num_layers * [hidden_size]
        layer_sizes = [2*hidden_size] + hidden_sizes
        linear_X = nn.Linear(len(config.Control_Col), hidden_size)
        self.net_type = net_type

        for i in range(1, len(layer_sizes)):
            modules_list.append(
                nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            )
            modules_list.append(nn.Tanh())
        modules_list.append(
            nn.Linear(layer_sizes[-1], hidden_size)
        )
        sequential_model = nn.Sequential(*modules_list)


        class MLPCell(nn.Module):
            def __init__(self, sequential_model, LX, x_size):
                super(MLPCell, self).__init__()
                self.sequential_model = sequential_model
                self.LX = LX
                self.x_size = x_size

            def forward(self, input):
                """

                :param input:(h, input)
                """
                x = input[:,:self.x_size]
                u = input[:, self.x_size:]
                return self.sequential_model(torch.cat([x, self.LX(u)], dim=1))

        self.ode_net = ODENet(MLPCell(sequential_model, linear_X, hidden_size), self.config.t_step,
                              self.config.interpolation)

