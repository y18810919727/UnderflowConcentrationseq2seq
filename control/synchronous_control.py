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
from control.model.synchronous import SynchronousController

# 载入pth
#state_dic = torch.load('./ckpt/lstm_ode_4_5/95.pth')
# 更新使用的仿真模型
state_dic = torch.load('./ckpt/rnn_ode_2_3_nobias/best.pth')


from models.ode import MyODE

# 与预测建模的实验公用config
net = MyODE(input_size=len(Target_Col+Control_Col),
            num_layers=config.num_layers, hidden_size=config.hidden_num, out_size=len(Target_Col), net_type=config.net_type)

net.load_state_dict(state_dic['net'])

# 自定义数据归一化工具
_mean, _var = state_dic['scaler_mean'], state_dic['scaler_var']
my_scaler = MyScaler(_mean, _var, Target_Col, Control_Col, config.controllable, config.uncontrollable)

thickener = Thickener(net, my_scaler, None)
quadratic_cost = QuadraticCost(fcn=net.fc)
quadratic_cost.y_target = my_scaler.scale_target(torch.FloatTensor(config.y_target))
synchronous_controller = SynchronousController(evn=thickener, scaler=my_scaler, dim_x_c=config.hidden_num+2, quadratic_cost=quadratic_cost)

synchronous_controller.train()