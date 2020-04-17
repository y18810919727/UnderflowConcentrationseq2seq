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


def cal_jac():
    x = np.arange(1, 3, 1)
    x = torch.from_numpy(x).reshape(len(x), 1)
    x = x.float()
    x.requires_grad = True

    w1 = torch.randn((2, 2), requires_grad=False)
    y = w1 @ x

    print(w1)
    jacT = torch.zeros(2, 2)
    for i in range(2):
        output = torch.zeros(2, 1)
        output[i] = 1.
        j = torch.autograd.grad(y, x, grad_outputs=output, retain_graph=True)
        a = j[0].shape
        b = jacT[:, i:i + 1].shape
        c = jacT[:, i].shape
        jacT[:, i:i + 1] = j[0]

# 载入pth

cal_jac()

state_dic = torch.load('./ckpt/lstm_ode_4_5/95.pth')

from models.ode import MyODE

# 与预测建模的实验公用config
net = MyODE(input_size=len(Target_Col+Control_Col),
            num_layers=config.num_layers, hidden_size=config.hidden_num, out_size=len(Target_Col), net_type=config.net_type)

net.load_state_dict(state_dic['net'])

# 自定义数据归一化工具
_mean, _var = state_dic['scaler_mean'], state_dic['scaler_var']
my_scaler = MyScaler(_mean, _var, Target_Col, Control_Col, config.controllable, config.uncontrollable)


thickener = Thickener(net, my_scaler, None)

# 随机生成控制量进行测试
u = torch.rand(1, 3)
nx, x_grad, f_u_grad = thickener.f(u)
print(nx)

# 定义效用计算器，计算效用值
quadratic_cost = QuadraticCost(net.fc)
utility = quadratic_cost.get_cost(nx, u)
print(utility)


