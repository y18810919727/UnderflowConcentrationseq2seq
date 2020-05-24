#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from control.thickener import Thickener
from custom_dataset import Control_Col, Target_Col
from config import args as config
from control.scaler import MyScaler
from control.cost_func import QuadraticCost
import time
from tensorboardX import SummaryWriter
from control.scaler import MyScaler as scaler
from models.model_generator import initialize_model
import common

if config.is_write > 0:
    writer = SummaryWriter(os.path.join('logs', 'ode_control', str(
        time.strftime("%Y%m%d%H%M%S", time.localtime())) + '__' + 'test_noise_new_model'))

# 载入pth

# state_dic = torch.load('./ckpt/lstm_ode_4_5/95.pth')
# state_dic = torch.load('./ckpt/rnn_ode_2_3_nobias/best.pth')
model_dir = 'rnn_ode_affine_2_3_h16'
model_name = 'best.pth'

# 载入pth
state_dic = torch.load(
    os.path.join('./ckpt', model_dir, model_name ))
assert common.parser_dir(model_dir, config)
print(config)
net = initialize_model(config)
net.load_state_dict(state_dic['net'])

# 自定义数据归一化工具
_mean, _var = state_dic['scaler_mean'], state_dic['scaler_var']
my_scaler = MyScaler(_mean, _var, Target_Col, Control_Col, config.controllable, config.uncontrollable)


thickener = Thickener(net, my_scaler, None)


quadratic_cost = QuadraticCost(net.fc)
quadratic_cost.y_target = my_scaler.scale_target(torch.FloatTensor(config.y_target))
step = 0
u = torch.FloatTensor([[0.2, 0.1, 0.2]])
while True:
    if step % 6000 == 1:
        u = torch.rand(1, 3)
    x, x_grad, f_u_grad = thickener.f(u)
    print('f_u_grad:'+str(f_u_grad.data.numpy().tolist()))
    # 定义效用计算器，计算效用值
    utility = quadratic_cost.get_cost(x, u)
    u_unscale = my_scaler.unscale_controllable(u)
    if config.is_write > 0:
        print('strp-'+str(step))
        writer.add_scalar('Critic/Reward', utility.data[0], step)
        # writer.add_scalar('Critic/Value', critic_value.data[0], step)

        for i in range(len(thickener.controllable_in_input_index)):
            writer.add_scalar('Actor/u' + str(i), u_unscale.data[0][i], step)

        y = quadratic_cost.fcn(x)
        y = my_scaler.unscale_target(y)
        writer.add_scalar('State/Concentration of underﬂow', y.data[0][0], step)
        writer.add_scalar('State/Height of mud layer', y.data[0][1], step)

        c_unscale = my_scaler.unscale_uncontrollable(thickener.c)
        writer.add_scalar('Noise/noise1', c_unscale.data[0][0], step)
        writer.add_scalar('Noise/noise2', c_unscale.data[0][1], step)
    step += 1



