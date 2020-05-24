#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import torch
import math
import os
import json

import torch
import argparse
from common import MyWriter
DATA_PATH = './data/res_all_selected_features_0.csv'
# DATA_PATH = './data/data_example.csv'


parser = argparse.ArgumentParser(description='seq2seq Data-driven Thickener Simulator')
parser.add_argument('--use_cuda', action='store_true', default=False)
parser.add_argument('--tb', action='store_true', default=False, help='write log to ./logs/save_dir')
parser.add_argument('--plt_visualize', action='store_true', default=False)
parser.add_argument('--loss_func', type=str, default='L2')
parser.add_argument('--net_type', type=str, default='rnn')
parser.add_argument('--algorithm', type=str, default='ode')
parser.add_argument('--hidden_num', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--look_back', type=int, default=160)
parser.add_argument('--look_forward', type=int, default=60)
parser.add_argument('--test_look_forward', type=int, default=60)
parser.add_argument('--sample_dis', type=int, default=10)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--save_dir', type=str, default='_basic')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--test_period', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--test_model', type=str, default='0')
parser.add_argument('--is_train', type=str, default=False)
parser.add_argument('--test_all', action='store_true', default=False)
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--nou', action='store_true', default=False)
parser.add_argument('--t_step', type=float, default=10.0/60)
parser.add_argument('--encode_rnn', type=str, default='rnn', help='rnn or lstm or GRU')

parser.add_argument('--rtol', type=str, default='1')
parser.add_argument('--atol', type=str, default='2')
parser.add_argument('--interpolation', type=str, default="quadratic", help="slinear, quadratic, cubic")
parser.add_argument('--begin', type=str, default="GRU_st", help="rnn_st, zero_st, learn_st")
parser.add_argument('--ode_method', type=str, default="dopri5", help="methods for computing ode")


parser.add_argument('--DATA_PATH', type=str, default='./data/res_all_selected_features_half.csv')
parser.add_argument('--data_half', action='store_true', default=False, help='double data separation')
parser.add_argument('--test_re', type=str, default='*')

parser.add_argument('--con_algorithm', type=str, default='synchronous')
parser.add_argument('--no_hidden_diff', action='store_true', default=False)
parser.add_argument('--min_future_length', type=int, default=30000)
parser.add_argument('--con_t_range', type=float, default=1000*10.0/60)
parser.add_argument('--controllable', type=list, default=['5','7','15'])
parser.add_argument('--uncontrollable', type=list, default=['4', '16'])
parser.add_argument('--con_batch_size', type=int, default=1)
parser.add_argument('--all_col', type=list, default=['1','4','5','7','11','15','16','17','18','19','20','21','22'])
parser.add_argument('--y_target', type=list, default=[63, 32])
parser.add_argument('--constant_noise', type=int, default=0)
parser.add_argument('--is_write', type=int, default=1)
parser.add_argument('--x_decode', type=int, default=1)
parser.add_argument('--phi_input', type=int, default=1)
parser.add_argument('--mpc_control', type=int, default=1)
parser.add_argument('--action_constraint', type=int, default=0)

args = parser.parse_args()

if args.epochs == 0:
    args.epochs = 1e8

if args.data_half:
    args.DATA_PATH = './data/res_all_selected_features_half.csv'

use_cuda = args.use_cuda if torch.cuda.is_available() else False
args.use_cuda = use_cuda

if 'ode' in args.algorithm:
    args.rtol = float('1e-' + args.rtol)
    args.atol = float('1e-' + args.atol)

if args.test_model == '0':
    args.is_train = True

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

args.save_dir = args.net_type + args.save_dir
if args.nou:
    args.save_dir = args.save_dir + '_nou'
if args.algorithm == 'ode':
    args.save_dir = args.save_dir + '_' + args.ode_method
if args.is_train and args.test_all:
    raise AttributeError('Can not test all in training phase.')


import time

if not os.path.exists('logs'):
    os.makedirs('logs')

if args.test_model == '0':
    args.tb_path = os.path.join('logs', args.save_dir, str(time.strftime("%Y%m%d%H%M%S", time.localtime())))
else:
    args.tb_path = os.path.join('expresults/logs', args.save_dir)




from sklearn.preprocessing import StandardScaler
import pandas as pd
data = pd.read_csv(args.DATA_PATH)
args.scaler = StandardScaler().fit(data)

from control.scaler import MyScaler
from custom_dataset import Target_Col, Control_Col
args.my_scaler = MyScaler(args.scaler.mean_, args.scaler.var_, args.all_col, Target_Col, Control_Col, args.controllable, args.uncontrollable)

