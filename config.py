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

parser = argparse.ArgumentParser(description='seq2seq Data-driven Thickener Simulater')
parser.add_argument('--use_cuda', action='store_true', default=False)
parser.add_argument('--tb_visualize', action='store_true', default=False, help='write log to ./logs/save_dir')
parser.add_argument('--plt_visualize', action='store_true', default=False)
parser.add_argument('--loss_func', type=str, default='L2')
parser.add_argument('--net_type', type=str, default='lstm')
parser.add_argument('--algorithm', type=str, default='diff')
parser.add_argument('--hidden_num', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--look_back', type=int, default=120)
parser.add_argument('--look_forward', type=int, default=60)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--save_dir', type=str, default='_basic')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--test_period', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--test_model', type=str, default='0')
parser.add_argument('--is_train', type=str, default=False)
parser.add_argument('--test_all', action='store_true', default=False)
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--t_range', type=float, default=10.0)
parser.add_argument('--rtol', type=str, default='3')
parser.add_argument('--atol', type=str, default='4')







args = parser.parse_args()

if args.epochs == 0:
    args.epochs = 1e8


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
if args.is_train and args.test_all:
    raise AttributeError('Can not test all in training phase.')


import time

if os.path.exists('logs'):
    os.makedirs('logs')
args.writer = MyWriter(
    save_path=os.path.join('logs', args.save_dir, str(time.strftime("%Y%m%d%H%M%S", time.localtime()))),
    is_write=args.tb_visualize
)
