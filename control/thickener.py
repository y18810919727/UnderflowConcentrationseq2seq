#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch.autograd import Variable
import pandas
from control.scaler import MyScaler
Target_Col =  ['11', '17']
Control_Col =  ['4','5','7','15','16']

from torchdiffeq import odeint
from common import col2Index
from decimal import Decimal
from config import args
import random

class Thickener():

    def __init__(self,
                 model,
                 scaler: MyScaler,
                 random_seed=None,
                 T= 1,
                 m=3,
                 batch_size=1,
                 config=None
                 ):

        """

        :param model: 建模实验中保存的模型
        :param scaler: 自定义的归一化工具
        :param random_seed: 随机化种子，会决定浓密机的初始状态和
        :param T: 仿真间隔
        :param m: 可控量维度
        :param batch_size: 控制实验中一般设为1
        """
        if random_seed is None:
            random_seed = random.randint(0, 1000000)
            print('random_seed:'+str(random_seed))
        self.random_seed = random_seed
        if config is None:
            from config import args as config
        config.min_future_length = 20000
        self.config = config

        # Time step
        self.T = T
        self.t = 0
        self.pi = 3.14159265359
        self.batch_size = batch_size

        self.rnn = model.rnn
        self.ode_net = model.ode_net
        self.m = m
        self.fcn = model.fc
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
        self.hidden_num = self.x.shape[1]

        # self.begin_index = [9680]

        print('begin_index:' + str(self.begin_index))

        self.c_u_seq = torch.stack(
            [torch.FloatTensor(self.scaled_data[ind:ind+config.min_future_length, self.input_index]) for ind in self.begin_index],
            dim=1
        )

        if args.constant_noise > 0:
            self.c_u_seq[:, :, self.uncontrollable_in_input_index] = torch.FloatTensor([0,0])

        self.c = self.c_u_seq[0][:, self.uncontrollable_in_input_index]


    def update_c_u_seq(self, u, is_pre = False):
        # 寻找当前生成c的下标
        if is_pre:
            self.input_position = max(int(self.t / self.config.t_step) - 1, 0)
        else:
            self.input_position = int(self.t / self.config.t_step)
        # 更新指定下标位置及相邻写一个位置的外部输入量u
        self.c_u_seq[self.input_position, :, self.controllable_in_input_index] = u
        self.c = self.c_u_seq[self.input_position][:, self.uncontrollable_in_input_index]

        if self.input_position + 1 < len(self.c_u_seq):
            self.c_u_seq[self.input_position + 1, :, self.controllable_in_input_index] = u


    def f(self, u):
        """
        根据自己生产的扰动量c以及输入控制量u，更新状态x并给出x的导数
        :param u: (bs, m) or (m,)
        :return: x(t+T), x对时间的导数
        """
        if u.shape is (self.m,):
            u = u.unsqueeze(dim=0)

        assert self.batch_size == u.shape[0]

        self.update_c_u_seq(u)

        # # 计算f_u_grad
        # ode_input = Variable(torch.cat((self.x, self.c_u_seq[self.input_position]), dim=1), requires_grad=True)
        # ode_ouput = self.ode_net.grad_module(ode_input)
        # if args.x_decode > 0:
        #     ode_ouput = self.fcn(ode_ouput)
        # jacT = torch.zeros(ode_input.shape[1], ode_ouput.shape[1])
        # for i in range(ode_ouput.shape[1]):
        #     gradients = torch.zeros(1, ode_ouput.shape[1])
        #     gradients[:, i] = 1
        #     j = torch.autograd.grad(ode_ouput, ode_input, grad_outputs=gradients, retain_graph=True)
        #     jacT[:, i:i + 1] = j[0].T
        # f_u_index = [i+self.hidden_num for i in self.controllable_in_input_index]
        # f_u_grad = jacT[f_u_index, :]

        # 更新ode求解模型中的控制输入序列

        def linear_fit(a, b, x):
            assert 0 <= x <= 1
            return a + (b-a)*x

        # 拿出t时刻顺时的外部变化量，用于计算x的顺时导数
        cur_c_u = linear_fit(
            self.c_u_seq[self.input_position],
            self.c_u_seq[min(self.input_position+1, len(self.c_u_seq) - 1)],
            (self.t - self.config.t_step * self.input_position) / self.config.t_step
        )
        t = torch.FloatTensor([self.t, self.t+self.T])

        self.last_x = self.x

        with torch.no_grad():
            # 计算导数
            dx_dt = self.ode_net.grad_module(torch.cat([self.x, cur_c_u], dim=1))
            # 计算t+T时刻的系统状态1
            from common import discrete_odeint
            x = discrete_odeint(self.ode_net, self.c_u_seq, self.x, t, rtol=self.config.rtol, atol=self.config.atol)[1]


        self.t = Decimal(str(self.t)) + Decimal(str(self.T))
        self.t = float(str(self.t))
        self.x = x

        if args.x_decode > 0:
            x_decode = self.fcn(x).data
            dx_dt = self.fcn(dx_dt).data
            return x_decode, dx_dt

        return x, dx_dt

    def f_pre(self, x, u):
        if u.shape is (self.m,):
            u = u.unsqueeze(dim=0)

        assert self.batch_size == u.shape[0]

        self.update_c_u_seq(u)

        def linear_fit(a, b, x):
            assert 0 <= x <= 1
            return a + (b-a)*x

        # 拿出t时刻顺时的外部变化量，用于计算x的顺时导数
        cur_c_u = linear_fit(
            self.c_u_seq[self.input_position],
            self.c_u_seq[min(self.input_position + 1, len(self.c_u_seq) - 1)],
            (self.t - self.config.t_step * self.input_position) / self.config.t_step
        )
        t = torch.FloatTensor([self.t, self.t + self.T])

        with torch.no_grad():
            # 计算导数
            f_input = torch.cat([x, cur_c_u], dim=1)
            # print(str(f_input.shape))
            dx_dt = self.ode_net.grad_module(f_input)
            # 计算t+T时刻的系统状态1
            from common import discrete_odeint
            x = discrete_odeint(self.ode_net, self.c_u_seq, x, t, rtol=self.config.rtol, atol=self.config.atol)[1]

        x_decode = self.fcn(x).data
        dx_dt = self.fcn(dx_dt).data
        return x_decode, dx_dt

    def initial_hidden_state(self, random_seed, scaled_data):


        np.random.seed(random_seed)
        # 定义batch_size 个随机起点
        begin_index = np.random.randint(0, self.length - self.config.look_back + 1 - self.config.min_future_length, self.batch_size)
        # begin_index = np.random.randint(0, 10, self.batch_size)

        # 拿出历史数据，准备使用rnn编码

        y = [torch.FloatTensor(scaled_data[ind:ind+self.config.look_back, self.target_index]) for ind in begin_index]
        u = [torch.FloatTensor(scaled_data[ind:ind+self.config.look_back, self.input_index]) for ind in begin_index]
        y = torch.stack(y,dim=1)
        u = torch.stack(u,dim=1)

        historical_series = torch.cat([u, y],dim=2)


        # 编码出初始状态
        with torch.no_grad():
            output, hn = self.rnn(historical_series)
        if type(hn) is tuple:
            hn = hn[0]

        # The shape of hn is (num_layers, batch_size, hidden_num ). Here, it assumes num_layers is equal to 1.

        return hn[0], begin_index+self.config.look_back-1




