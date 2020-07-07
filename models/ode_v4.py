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




class MyODEV4(MyODE):

    def __init__(self, input_size, num_layers, hidden_size, out_size, config, net_type='lstm'):

        super(MyODE, self).__init__(input_size, num_layers, hidden_size, out_size, config, net_type)

        linear_X = nn.Linear(len(config.Control_Col), hidden_size)
        self.GRU_cell = nn.GRUCell(hidden_size, hidden_size)

        class MyGRUCell(nn.Module):
            def __init__(self, LX, x_size):
                super(MyGRUCell, self).__init__()
                self.LX = LX
                self.x_size = x_size

                self.GRU_cell = nn.GRUCell(x_size, x_size)

            def forward(self, input):
                """

                :param input:(h, input)
                """
                x = input[:, :self.x_size]
                u = input[:, self.x_size:]
                return self.GRU_cell(self.LX(u), x)*0.1 - x

        self.ode_net = ODENet(MyGRUCell(linear_X, hidden_size), self.config.t_step,
                              self.config.interpolation)

