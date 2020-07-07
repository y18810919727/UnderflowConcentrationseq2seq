import numpy as np
import math
import os
import json

import torch
from control.thickener import Thickener
from control.scaler import MyScaler
from control.cost_func import QuadraticCost
from control.model.synchronous import SynchronousController
import common

#state_dic = torch.load('./ckpt/lstm_ode_4_5/95.pth')

# 更新使用的仿真模型

from config import args as config

Control_Col = config.Control_Col
Target_Col = config.Target_Col
from models.model_generator import initialize_model

#model_dir = 'rnn_ode_2_3_h32'
model_dir = 'rnn_ode_affine_2_3_h16'
model_name = 'best.pth'

# 载入pth
state_dic = torch.load(
    os.path.join('./ckpt', model_dir, model_name ))
assert common.parser_dir(model_dir, config)
print(config)
net = initialize_model(config)
net.load_state_dict(state_dic['net'])


# 与预测建模的实验公用config
# net = MyODE(input_size=len(Target_Col+Control_Col),
#             num_layers=config.num_layers, hidden_size=config.hidden_num, out_size=len(Target_Col), net_type=config.net_type)
#
# net.load_state_dict(state_dic['net'])

# 自定义数据归一化工具
_mean, _var = state_dic['scaler_mean'], state_dic['scaler_var']
my_scaler = MyScaler(_mean, _var, Target_Col, Control_Col, config.controllable, config.uncontrollable)

thickener = Thickener(net, my_scaler, None)
quadratic_cost = QuadraticCost(fcn=net.fc)
quadratic_cost.last_u = torch.FloatTensor([[0, 0, 0]])
quadratic_cost.y_target = my_scaler.scale_target(torch.FloatTensor(config.y_target))
synchronous_controller = SynchronousController(evn=thickener, scaler=my_scaler, quadratic_cost=quadratic_cost)

synchronous_controller.train()