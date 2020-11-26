#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
class Interpolation(nn.Module):
    def __init__(self, t_seq, x, method='cubic'):
        """
        :param t: 1D array with shape (len,)
        :param x: (len, batch_size, x_size)
        """
        super().__init__()
        assert t_seq.shape[0] == x.shape[0]
        self.t_seq = t_seq
        self.x = x
        self.max_t = t_seq[-1]
        self.min_t = t_seq[0]
        self.len = t_seq.shape[0]
        self.method = method
        self.batch_size, self.x_size = x.shape[1:]
        self.order = self.make_order()

    def make_order(self):
        if self.method == 'cubic':
            order = 3
        elif self.method == 'quadratic':
            order = 2
        elif self.method == 'slinear':
            order = 1
        elif self.method == 'zero':
            order = 0
        else:
            raise NotImplemented('Interpolation method - {} is not been implemented'.format(self.method))
        return min(order, self.len-1)

    def expand_t(self, t):
        return torch.stack([t ** o for o in range(self.order + 1)], dim=-1)

    def find_nearest_integer_points(self, x_target):
        eps = 1e-9
        #x_target = (t - self.min_t) / (self.max_t - self.min_t) * (self.len - 1)
        x1 = torch.LongTensor([x_target + eps]).squeeze()
        x_target = torch.LongTensor([x_target]).squeeze()
        x1 -= min(x1, max(x1 + self.order - self.len + 1, 0))
        # while x1>=1 and x_right >= self.len:
        #     x1 -= 1
        #     x_right = x1 + self.order
        if self.order == 0:
            if x_target - x1 > x1 + 1 - x_target:
                x1 = x1 + 1
        xs = torch.linspace(x1, x1 + self.order, self.order + 1).to(self.x.device)
        # import pdb
        # pdb.set_trace()
        ps = torch.stack([self.x[int(x_ind)].contiguous().reshape(self.batch_size * self.x_size) for x_ind in xs], dim=0)
        return xs, ps

    def forward(self, t):
        x_target = (t-self.min_t)/(self.max_t - self.min_t) * (self.len - 1)

        xs, ps = self.find_nearest_integer_points(x_target)
        xs_mat = self.expand_t(xs)
        assert xs_mat.shape == (self.order + 1, self.order + 1)
        assert ps.shape == (self.order + 1, self.batch_size*self.x_size)
        params = xs_mat.inverse().matmul(ps)

        result = self.expand_t(x_target).matmul(params).view(self.batch_size, self.x_size)
        return result


class EmptyInterpolation(nn.Module):
    def __init__(self):
        super(EmptyInterpolation, self).__init__()
        pass

    def forward(self, t):
        raise AttributeError('This is an empty interpolation. Please specify the interpolaiton module for instance of OdeSystem.')
