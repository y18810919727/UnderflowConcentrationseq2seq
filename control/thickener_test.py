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

# 注意1：这里提供了三个模型，一个affine的，两个普通的，旧的模型不能用了
model_dic = {
    'model1': './ckpt/rnn_ode_3_4_h32_cubic_transform_dopri5/best.pth',
    'model2': './ckpt/GRU_ode_3_4_h32_cubic_dopri5/best.pth',
    'model3': './ckpt/rnn_ode_affine_3_4_cubic_transform/best.pth'
}

test = 'model1'
#state_dic = torch.load('./ckpt/rnn_ode_3_4_h32_cubic_transform_dopri5/best.pth')
#state_dic = torch.load('./ckpt/GRU_ode_3_4_h32_cubic_dopri5/best.pth')
state_dic = torch.load(model_dic[test])

if config.is_write > 0:
    if config.constant_noise > 0:
        writer = SummaryWriter(os.path.join('logs', 'model_test', str(
            time.strftime("%Y%m%d%H%M%S", time.localtime())) + '__' + 'test_new_' + test))
    else:
        writer = SummaryWriter(os.path.join('logs', 'model_test', str(
            time.strftime("%Y%m%d%H%M%S", time.localtime())) + '__' + 'test_noise_new_' + test))

from models.model_generator import initialize_model

# 保存的pth文件中直接记录着当时训练模型时的config字典
model_config = state_dic['config']

net = initialize_model(config=model_config)

net.load_state_dict(state_dic['net'])
net.ode_net.interpolation_kind = 'slinear'
# 自定义数据归一化工具
_mean, _var = state_dic['scaler_mean'], state_dic['scaler_var']

print(str(_mean.tolist()))
print(str(_var.tolist()))

# 注意2: MyScaler构造函数增加了一参数all_col
my_scaler = MyScaler(_mean, _var, model_config.all_col,  Target_Col, Control_Col, config.controllable, config.uncontrollable)

# 注意3: Thickener 的构造函数中需要添加config参数
thickener = Thickener(net, my_scaler, random_seed=781059, config=model_config)

step = 0
u = torch.FloatTensor([[0.2, 0.1, 0.2]])
while True:
    print('step-'+str(step))
    # if step % 6000 == 1:
    #     u = torch.rand(1, 3)
    x, x_grad = thickener.f(u)
    u_unscale = thickener.scaler.unscale_controllable(u)
    
    if config.is_write > 0:
        for i in range(len(thickener.controllable_in_input_index)):
            writer.add_scalar('Actor/u' + str(i), u_unscale.data[0][i], step)
        y = thickener.scaler.unscale_target(x)
        writer.add_scalar('State/Concentration of underﬂow', y.data[0][0], step)
        writer.add_scalar('State/Height of mud layer', y.data[0][1], step)

        c_unscale = my_scaler.unscale_uncontrollable(thickener.c)
        writer.add_scalar('Noise/noise1', c_unscale.data[0][0], step)
        writer.add_scalar('Noise/noise2', c_unscale.data[0][1], step)
    step += 1
    if step > 1000:
        break



