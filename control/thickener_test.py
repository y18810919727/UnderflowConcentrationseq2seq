#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from control.thickener import Thickener
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
    'model3': './ckpt/rnn_ode_affine_3_4_cubic_transform/best.pth',
    'model4': './ckpt/rnn_ode_affine_3_4_cubic_full/best.pth',
    'model5': './ckpt/rnn_ode_affine_3_4_cubic_transform_full/best.pth',
    'model6': '/ckpt/GRU_sta_euler_b20_f15/best.pth'
}


def get_data_path():
    state_dic = torch.load(model_dic['model4'])
    return state_dic['config'].DATA_PATH

def model_test():
    test = 'model4'
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
    model_config.DATA_PATH = get_data_path()
    Control_Col = model_config.Control_Col
    Target_Col = model_config.Target_Col

    net = initialize_model(config=model_config)

    if 'net' in state_dic.keys():
        net.load_state_dict(state_dic['net'])
    else:
        net.load_state_dict(state_dic['net15'])
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
        if step > 3000:
            break

def model_test_short_time():

    state_dic1 = torch.load(model_dic['model4'])
    state_dic2 = torch.load(model_dic['model5'])

    u = torch.FloatTensor([[0.2, 0.1, 0.2]])
    from models.model_generator import initialize_model

    # 保存的pth文件中直接记录着当时训练模型时的config字典
    model_config1 = state_dic1['config']
    Control_Col1 = model_config1.Control_Col
    Target_Col1 = model_config1.Target_Col
    net1 = initialize_model(config=model_config1)
    net1.load_state_dict(state_dic1['net'])
    net1.ode_net.interpolation_kind = 'slinear'
    # 自定义数据归一化工具
    _mean1, _var1 = state_dic1['scaler_mean'], state_dic1['scaler_var']
    my_scaler1 = MyScaler(_mean1, _var1, model_config1.all_col, Target_Col1, Control_Col1, config.controllable,
                         config.uncontrollable)

    model_config2 = state_dic2['config']
    model_config2.DATA_PATH = model_config1.DATA_PATH
    Control_Col2 = model_config2.Control_Col
    Target_Col2 = model_config2.Target_Col
    net2 = initialize_model(config=model_config2)
    net2.load_state_dict(state_dic2['net'])
    net2.ode_net.interpolation_kind = 'slinear'
    # 自定义数据归一化工具
    _mean2, _var2 = state_dic1['scaler_mean'], state_dic1['scaler_var']
    my_scaler2 = MyScaler(_mean2, _var2, model_config1.all_col, Target_Col2, Control_Col2, config.controllable,
                          config.uncontrollable)
    diff_value_list = []
    for i in range(20):
        thickener1 = Thickener(net1, my_scaler1, random_seed=None, config=model_config1)
        thickener2 = Thickener(net2, my_scaler2, random_seed=thickener1.random_seed, config=model_config2)

        x1, dxdt1 = thickener1.f(u)
        x2, dxdt2 = thickener2.f(u)
        y1 = thickener1.scaler.unscale_target(x1)
        y2 = thickener2.scaler.unscale_target(x2)
        diff_value_list.append(y1.data[0][0].numpy().tolist() - y2.data[0][0].numpy().tolist())
    print(str(diff_value_list))
if __name__ == '__main__':
    model_test_short_time()