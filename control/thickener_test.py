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

#cal_jac()


# 注意1：这里提供了三个模型，一个affine的，两个普通的，旧的模型不能用了
#state_dic = torch.load('./ckpt/rnn_ode_3_4_h32_cubic_transform_dopri5/best.pth')
#state_dic = torch.load('./ckpt/GRU_ode_3_4_h32_cubic_dopri5/best.pth')
state_dic = torch.load('./ckpt/rnn_ode_affine_3_4_cubic_transform/best.pth')


from models.model_generator import initialize_model

# 保存的pth文件中直接记录着当时训练模型时的config字典
model_config = state_dic['config']

net = initialize_model(config=model_config)

# net = MyODE(input_size=len(Target_Col+Control_Col),
#             num_layers=config.num_layers, hidden_size=config.hidden_num, out_size=len(Target_Col), net_type=config.net_type)
net.load_state_dict(state_dic['net'])
net.ode_net.interpolation_kind = 'slinear'

# 自定义数据归一化工具
_mean, _var = state_dic['scaler_mean'], state_dic['scaler_var']

# 注意2: MyScaler构造函数增加了一参数all_col
my_scaler = MyScaler(_mean, _var, model_config.all_col,  Target_Col, Control_Col, config.controllable, config.uncontrollable)


# 注意3: Thickener 的构造函数中需要添加config参数
thickener = Thickener(net, my_scaler, None, config=model_config)

# 随机生成控制量进行测试
u = torch.rand(1, 3)
nx, x_grad, f_u_grad = thickener.f(u)
print(nx)

# 定义效用计算器，计算效用值
quadratic_cost = QuadraticCost(fcn=net.fc)
quadratic_cost.y_target = my_scaler.scale_target(torch.FloatTensor(config.y_target))
utility = quadratic_cost.get_cost(nx, u)
print(utility)


