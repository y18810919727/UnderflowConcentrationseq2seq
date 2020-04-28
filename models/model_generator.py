#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

from custom_dataset import Target_Col, Control_Col


def initialize_model(config):

    if config.algorithm == 'diff':
        from models.diff import DiffNet as SeriesNet
    elif config.algorithm == 'ode':
        from models.ode import MyODE as SeriesNet
    elif config.algorithm == 'ode_affine':
        from models.ode_affine_linear import MyODEAffine as SeriesNet
    elif config.algorithm == 'hidden_rnn':
        from models.hidden_rnn import HiddenRNN as SeriesNet
    elif config.algorithm.startswith('RK'):

        from models.RK import RK as SeriesNet
    else:
        raise AttributeError

    net = SeriesNet(input_size=len(Target_Col+Control_Col),
                    num_layers=config.num_layers, hidden_size=config.hidden_num, out_size=len(Target_Col), net_type=config.net_type)
    return net
