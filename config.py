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
action_1 = [-1.76, 1.48]  # 3.24
action_2 = [-2.98, 0.66]  # 3.64
action_3 = [-1.39, 3.78]  # 5.03

parser = argparse.ArgumentParser(description='seq2seq Data-driven Thickener Simulator')
parser.add_argument('--use_cuda', action='store_true', default=False)
parser.add_argument('--tb', action='store_true', default=False, help='write log to ./logs/save_dir')
parser.add_argument('--plt_visualize', action='store_true', default=False)
parser.add_argument('--loss_func', type=str, default='L2')
parser.add_argument('--net_type', type=str, default='rnn')
parser.add_argument('--algorithm', type=str, default='ode')
parser.add_argument('--hidden_num', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--look_back', type=int, default=20)
parser.add_argument('--look_forward', type=int, default=15)
parser.add_argument('--test_look_forward', type=list, default=[15, 50, 125])
#parser.add_argument('--test_look_forward', type=list, default=[60, 200, 1000])


parser.add_argument('--cmp_length', type=str, default='15')
parser.add_argument('--sample_dis', type=int, default=1)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--test_period', type=int, default=20)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--test_model', type=str, default='0')
parser.add_argument('--is_train', type=str, default=False)
parser.add_argument('--test_all', action='store_true', default=False)
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--nou', action='store_true', default=False)
parser.add_argument('--t_step', type=float, default=0.1666667)
parser.add_argument('--encoder_rnn', type=str, default='RNN', help='RNN or LSTM or GRU or zero_st or learn_st')

parser.add_argument('--data_inv', action='store_true', default=False)
parser.add_argument('--rtol', type=str, default='4')
parser.add_argument('--atol', type=str, default='5')
parser.add_argument('--interpolation', type=str, default="quadratic", help="zero, slinear, quadratic, cubic")
parser.add_argument('--begin', type=str, default="GRU_st", help="rnn_st, zero_st, learn_st")
parser.add_argument('--ode_method', type=str, default="dopri5", help="methods for computing ode")
parser.add_argument('--stationary', action='store_true', help="the property of ode system")
parser.add_argument('--adjoint', action='store_true',
                    help="Choose to use adjoint sensitivity method  or not to backward ode-net")
parser.add_argument('--max_length_encoder', type=int, default=80)


#parser.add_argument('--DATA_PATH', type=str, default='./data/res_all_selected_features_half.csv')
parser.add_argument('--DATA_PATH', type=str, default='./data/res_all_selected_features_0.csv')
parser.add_argument('--data_choice', type=str, default='quarter')
parser.add_argument('--test_re', type=str, default='*')
parser.add_argument('--plot', type=str, default='')

parser.add_argument('--con_algorithm', type=str, default='synchronous')
parser.add_argument('--no_hidden_diff', action='store_true', default=False)
parser.add_argument('--min_future_length', type=int, default=30000)
parser.add_argument('--con_t_range', type=float, default=1000*10.0/60)
parser.add_argument('--dataset_name', type=str, default='thickener', help='thickener or cstr')
parser.add_argument('--Target_Col', type=list, default=['11', '17'])
parser.add_argument('--Control_Col', type=list, default=['4','5','7','15','16'])
parser.add_argument('--controllable', type=list, default=['5','7','15'])
parser.add_argument('--uncontrollable', type=list, default=['4', '16'])
parser.add_argument('--all_col', type=list, default=['1','4','5','7','11','15','16','17','18','19','20','21','22'])

parser.add_argument('--con_batch_size', type=int, default=1)
parser.add_argument('--y_target', type=list, default=[66, 32])
parser.add_argument('--constant_noise', type=int, default=0)
parser.add_argument('--is_write', type=int, default=1)
parser.add_argument('--x_decode', type=int, default=1)
parser.add_argument('--phi_input', type=int, default=1)
parser.add_argument('--mpc_control', type=int, default=1)
parser.add_argument('--action_constraint', type=int, default=0)
parser.add_argument('--noise_narrow', type=int, default=1)
parser.add_argument('--cal_u', type=int, default=0)
parser.add_argument('--noise_pre', type=int, default=1)


args = parser.parse_args()
if args.dataset_name == 'cstr':
    args.Target_Col = ['1', '2']
    args.Control_Col = ['0']
    args.all_col = ['0', '1', '2']
    args.controllable = ['0']
    args.uncontrollable = []
    args.DATA_PATH = './data/cstr.csv'
    args.save_dir = 'cstr'+args.save_dir
else:
    if args.data_choice == 'half':
        args.DATA_PATH = './data/res_all_selected_features_half.csv'
        # args.look_back = 40
        # args.look_forward = 30
        # args.test_look_forward = [30, 100, 250]
        # args.cmp_length = 30
    elif args.data_choice == 'quarter':
        args.DATA_PATH = './data/res_all_selected_features_quarter.csv'
        # args.look_back = 20
        # args.look_forward = 15
        # args.test_look_forward = [15, 50, 125]
        # args.cmp_length = 15


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

args.save_dir = args.net_type + ('' if args.save_dir == '' else '_') + args.save_dir
if args.nou:
    args.save_dir = args.save_dir + '_nou'
if args.is_train and args.test_all:
    raise AttributeError('Can not test all in training phase.')
if args.algorithm == 'ode':
    if args.stationary:
        args.save_dir = args.save_dir + '_sta'
    else:
        args.save_dir = args.save_dir + '_nonsta'
    args.save_dir = args.save_dir + '_' + args.ode_method
args.save_dir = args.save_dir + '_b' + str(args.look_back) + '_f' + str(args.look_forward)
import time

if not os.path.exists('logs'):
    os.makedirs('logs')

if args.test_model == '0':

    save_path = os.path.join('logs', args.save_dir)
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path, ignore_errors=True)
    args.tb_path = os.path.join('logs', args.save_dir, str(time.strftime("%Y%m%d%H%M%S", time.localtime())))
else:
    args.tb_path = os.path.join('expresults/logs', args.save_dir)




from sklearn.preprocessing import StandardScaler
import pandas as pd
data = pd.read_csv(args.DATA_PATH)
args.scaler = StandardScaler().fit(data)

from control.scaler import MyScaler
Target_Col = args.Target_Col
Control_Col = args.Control_Col

# import pdb
# pdb.set_trace()
args.my_scaler = MyScaler(args.scaler.mean_, args.scaler.var_, args.all_col, Target_Col, Control_Col, args.controllable, args.uncontrollable)

