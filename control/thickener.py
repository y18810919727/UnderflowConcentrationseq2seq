#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from config import args as config
import pandas
from control.scaler import MyScaler
from custom_dataset import Target_Col
from custom_dataset import Control_Col
from torchdiffeq import odeint

from common import col2Index

class Thickener():

    def __init__(self,
                 model,
                 scaler: MyScaler,
                 random_seed=None,
                 T=0.01,
                 m=3,
                 batch_size=1
                 ):

        """

        :param model: 建模实验中保存的模型
        :param scaler: 自定义的归一化工具
        :param random_seed: 随机化种子，会决定浓密机的初始状态和
        :param T: 仿真间隔
        :param m: 可控量维度
        :param batch_size: 控制实验中一般设为1
        """

        self.random_seed = random_seed

        # Time step
        self.T = T
        self.t = 0
        self.batch_size = batch_size

        self.rnn = model.rnn
        self.ode_net = model.ode_net
        self.m = m

        self.scaler = scaler

        ori_data = np.array(pandas.read_csv(config.DATA_PATH))

        self.scaled_data = self.scaler.scale_all(ori_data)
        self.length = len(self.scaled_data)

        # 寻找在原始数据集中，目标量的下标
        self.target_index = col2Index(config.all_col, Target_Col)
        # 寻找在原始数据集中，外部变化量的下标
        self.input_index = col2Index(config.all_col, Control_Col)

        # 寻找在原始数据集中，外部变化量中可控那部分的下标
        self.controllable_index = col2Index(config.all_col, config.controllable)
        # 寻找在原始数据集中，外部变化量中不可控那部分的下标
        self.uncontrollable_index = col2Index(config.all_col, config.uncontrollable)

        # 寻找在外部变化量中，可控那部分的下标
        self.controllable_in_input_index = col2Index(Control_Col, config.controllable)
        # 寻找在外部变化量中，不可控那部分的下标
        self.uncontrollable_in_input_index = col2Index(Control_Col, config.uncontrollable)


        # x: (batch_size, hidden_num),
        # begin_index: (batch_size), 记录位置用于后续生成外部噪音c
        self.x, self.begin_index = self.initial_hidden_state(self.random_seed, self.scaled_data)

        self.c_u_seq = torch.stack(
            [torch.FloatTensor(self.scaled_data[ind:ind+config.min_future_length, self.input_index]) for ind in self.begin_index],
            dim=1
        )


    def f(self, u):
        """
        根据自己生产的扰动量c以及输入控制量u，更新状态x并给出x的导数
        :param u: (bs, m) or (m,)
        :return: x(t+T), x对时间的导数
        """

        if u.shape is (self.m,):
            u = u.unsqueeze(dim=0)

        assert self.batch_size == u.shape[0]

        # 寻找当前生成c的下标
        input_position = int(self.t / config.t_step)

        # 更新指定下标位置及相邻写一个位置的外部输入量u
        self.c_u_seq[input_position, :, self.controllable_in_input_index] = u
        if input_position + 1 < len(self.c_u_seq):
            self.c_u_seq[input_position + 1, :, self.controllable_in_input_index] = u

        # 更新ode求解模型中的控制输入序列
        self.ode_net.set_u(self.c_u_seq)

        def linear_fit(a, b, x):
            assert 0 <= x <= 1
            return a + (b-a)*x

        # 拿出t时刻顺时的外部变化量，用于计算x的顺时导数
        cur_c_u = linear_fit(
            self.c_u_seq[input_position],
            self.c_u_seq[min(input_position+1, len(self.c_u_seq) - 1)],
            (self.t - config.t_step * input_position) / config.t_step
        )
        t = torch.FloatTensor([self.t, self.t+self.T])
        with torch.no_grad():
            # 计算导数
            dx_dt = self.ode_net.grad_module(torch.cat([self.x, cur_c_u], dim=1))
            # 计算t+T时刻的系统状态1
            self.x = odeint(self.ode_net, self.x, t, rtol=config.rtol, atol=config.atol)[1]

        self.t += self.T
        return self.x, dx_dt


    def initial_hidden_state(self, random_seed, scaled_data):


        np.random.seed(random_seed)

        # 定义batch_size 个随机起点
        begin_index = np.random.randint(0, self.length - config.look_back + 1 - config.min_future_length, self.batch_size)

        # 拿出历史数据，准备使用rnn编码
        y = [torch.FloatTensor(scaled_data[ind:ind+config.look_back, self.target_index]) for ind in begin_index]
        u = [torch.FloatTensor(scaled_data[ind:ind+config.look_back, self.input_index]) for ind in begin_index]
        y = torch.stack(y,dim=1)
        u = torch.stack(u,dim=1)

        historical_series = torch.cat([u, y],dim=2)


        # 编码出初始状态
        with torch.no_grad():
            output, hn = self.rnn(historical_series)
        if type(hn) is tuple:
            hn = hn[0]

        # The shape of hn is (num_layers, batch_size, hidden_num ). Here, it assumes num_layers is equal to 1.

        return hn[0], begin_index+config.look_back-1



