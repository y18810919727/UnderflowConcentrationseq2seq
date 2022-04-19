#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch



def initialize_model(config):
    from models.mimo import MIMO

    if config.algorithm == 'ode':
        net = MIMO(k_in=len(config.Control_Col),k_out=len(config.Target_Col), k_state=config.hidden_num, solver=config.ode_method,
             stationary=config.stationary, interpolation=config.interpolation, encoder_net_type=config.encoder_rnn,
             net_type=config.net_type, adjoint=config.adjoint, ut=config.t_step)
    elif config.algorithm == 'seq2seq':
        from models.seq2seq import AttentionSeq2Seq
        net = AttentionSeq2Seq(k_in=len(config.Control_Col), k_out=len(config.Target_Col),
                               k_state=config.hidden_num, max_length=config.max_length_encoder,
                               num_layers=config.num_layers)
    elif config.algorithm == 'ss':
        from models.state_space import StateSpace
        net = StateSpace(k_in=len(config.Control_Col), k_out=len(config.Target_Col),
                               k_state=config.hidden_num, max_length=config.max_length_encoder,
                               )

    # if config.algorithm == 'diff':
    #     from models.diff import DiffNet as SeriesNet
    # elif config.algorithm == 'ode':
    #     if config.ode_version == '1':
    #         from models.ode import MyODE as SeriesNet
    #     elif config.ode_version == '2':
    #         from models.ode_v2 import MyODEV2 as SeriesNet
    #     else:
    #         print(config.ode_version + 'is not defined')
    #
    #
    # elif config.algorithm == 'ode_affine':
    #     from models.ode_affine_linear import MyODEAffine as SeriesNet
    # elif config.algorithm == 'hidden_rnn':
    #     from models.hidden_rnn import HiddenRNN as SeriesNet
    # elif config.algorithm.startswith('RK'):
    #
    #     from models.RK import RK as SeriesNet
    # else:
    #     raise AttributeError
    #
    # net = SeriesNet(input_size=len(config.Target_Col+config.Control_Col),
    #                 num_layers=config.num_layers, hidden_size=config.hidden_num, out_size=len(config.Target_Col),
    #                 config=config, net_type=config.net_type)
    return net
