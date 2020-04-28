#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from control.scaler import MyScaler
from config import args as config

class CostFuncBase:
    def __init__(self, fcn, config=None):

        self.config = config
        self.fcn = fcn

        for key, value in config.items():
            vars(self)[key] = value

    def get_cost(self, x, u):
        raise NotImplementedError

    def solve_partial_equation(self, w, df_du_func, last_u=None):
        raise NotImplementedError


class QuadraticCost(CostFuncBase):

    def __init__(self, fcn, config=None):
        """

        :param fcn:
        :param scaler:
        :param config: The setting of parameters should be suit for real data, not for normalized data.
        """
        if config is None:
            # config = {
            #     'Q': torch.diag(torch.FloatTensor([10.0, 2.0])),
            #     'R': torch.diag(torch.FloatTensor([4.75, 9.2e-5, 8.2e-3])),
            #     'y_target': torch.FloatTensor([62.0, 45.0]),
            #     'u_mid': torch.FloatTensor([4, 1550, 80]),
            #     'n': 2,
            #     'm': 3,
            #     'EPS': 1e-7,
            # }

            # 03.29更新：直接拿归一化的数据计算cost更方便，跑出实验后做曲线可视化时再反归一化回来
            # 这些参数我目前是随机定的，可以根据实验结果微调
            config = {
                'Q': torch.diag(torch.FloatTensor([10.0, 0.001])),
                'R': torch.diag(torch.FloatTensor([0.001, 0.001, 0.001])),
                'u_mid': torch.FloatTensor([0, 0, 0]),
                'n': 2,
                'm': 3,
            }
        super(QuadraticCost, self).__init__(fcn, config)


    def get_cost(self, x, u):
        if u.shape == (self.m,):
            u.unsqueeze(dim=0)
        if len(x.shape) == 1:
            x.unsqueeze(dim=0)
        assert x.shape[0] == u.shape[0] and u.shape[1] == self.m

        y = self.fcn(x)

        y_det = y-self.y_target
        u_det = u-self.u_mid
        y_cost = torch.sum(y_det @ self.Q @ y_det.T, dim=1)
        u_cost = torch.sum(u_det.matmul(self.R).matmul(u_det.T), dim=1)
        print('y_cost:'+str(y_cost.data[0])+'; u_cost:'+str(u_cost.data[0]))
        return y_cost + u_cost




#Todo
#class IntegralCost(CostFuncBase):




