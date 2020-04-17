#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
import torch
import time
import itertools
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from control.thickener import Thickener
from config import args as config
from control.cost_func import QuadraticCost
from control.scaler import MyScaler


class SynchronousController:

    def __init__(self,
                 evn: Thickener,
                 quadratic_cost: QuadraticCost,
                 scaler: MyScaler,
                 gpu_id=0,
                 traning_rounds=10000,
                 iter_rounds=500,
                 actor_err_limit=0.01,
                 critic_err_limit=0.01,
                 actor_lr=0.01,
                 critic_lr=0.01,
                 dim_x_c=18,

                 step_max=30000
                 ):

        self.evn = evn
        self.quadratic_cost = quadratic_cost
        self.scaler = scaler
        self.gpu_id = gpu_id
        self.training_rounds = traning_rounds
        self.iter_rounds = iter_rounds
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_err_limit = actor_err_limit
        self.critic_err_limit = critic_err_limit
        self.step_max = step_max

        self.dim_x_c = dim_x_c

        # self.W_size = int(self.dim_x_c * (self.dim_x_c+1) / 2)
        self.W_size = self.dim_x_c ** 2
        self.W_1 = torch.rand(self.W_size, 1)
        self.W_2 = torch.rand(self.W_size, 1)

        f_value = self.W_size * [0.1]

        self.F_1 = torch.unsqueeze(torch.FloatTensor(f_value), dim=1)
        self.F_2 = torch.FloatTensor([0.1])
        parameter_name = 'y_target=' + str(config.y_target)
        self.writer = SummaryWriter(os.path.join('logs', 'ode_control', parameter_name + '__' + str(
            time.strftime("%Y%m%d%H%M%S", time.localtime()))))
        print(parameter_name)

    def phi(self, x):
        """
        多项式向量
        :param x: 输入
        :return:
        """
        assert x.shape[1] == self.dim_x_c
        x = Variable(x, requires_grad=True)
        phi_value = x.T @ x
        phi_value = phi_value.view(self.W_size)
        jacT = torch.zeros(self.dim_x_c, self.W_size)
        for i in range(self.W_size):
            output = torch.zeros(self.W_size)
            output[i] = 1
            j = torch.autograd.grad(phi_value, x, grad_outputs=output, retain_graph=True)
            jacT[:, i:i + 1] = j[0].T
        return phi_value, jacT[:16, :]

    def df_du_fun(self, u, w):
        """
        计算 f对u的导数 * w
        :param u:
        :param w: phi @ W_2
        :return: f_u_grad
        """
        u = Variable(u, requires_grad=True)
        self.evn.update_c_u_seq(u)
        ode_input = Variable(torch.cat((self.evn.last_x, self.evn.c_u_seq[self.evn.input_position]), dim=1),
                             requires_grad=True)
        ode_ouput = self.evn.ode_net.grad_module(ode_input)
        j = torch.autograd.grad(ode_ouput, ode_input, grad_outputs=w.T, retain_graph=True)
        f_u_index = [i + config.hidden_num for i in self.evn.controllable_in_input_index]
        f_u_grad = j[0][:, f_u_index]
        return f_u_grad

    def train(self):

        u = torch.rand(1, 3)
        for step in range(self.step_max):
            x, x_grad, f_u_grad = self.evn.f(u)

            utility = self.quadratic_cost.get_cost(self.evn.last_x, u)
            phi, phi_x_grad = self.phi(torch.cat((x, self.evn.c), dim=1))
            critic_value = self.W_1.T @ phi
            u = self.quadratic_cost.solve_partial_equation(last_u=u, w=phi_x_grad @ self.W_2,
                                                           df_du_func=self.df_du_fun)
            # cost_list.append(utility.data.numpy().tolist())
            # c_list.append(critic_value.data.numpy().tolist())
            # u_list.append(u.data.numpy().tolist())

            self.writer.add_scalar('utility', utility.data[0], step)
            self.writer.add_scalar('critic_value', critic_value.data[0], step)
            u_unscale = self.scaler.unscale_controllable(u)
            for i in range(len(self.evn.controllable_in_input_index)):
                self.writer.add_scalar('u' + str(i), u_unscale.data[0][i], step)
            y = self.quadratic_cost.fcn(x)
            y = self.scaler.unscale_target(y)
            self.writer.add_scalar('Concentration of underﬂow', y.data[0][0], step)
            self.writer.add_scalar('Height of mud layer', y.data[0][1], step)

            # 参数更新
            sigma = phi_x_grad.T @ x_grad.T
            m = sigma / (sigma.T @ sigma + 1) ** 2
            D1 = phi_x_grad.T @ f_u_grad.T @ self.quadratic_cost.config['R'].inverse() @ f_u_grad @ phi_x_grad
            self.W_2 = self.W_2 - self.actor_lr * self.evn.T * (
                    self.F_2 * self.W_2 - self.F_1 @ sigma.T @ self.W_1 - D1 @ self.W_2 @ m.T @ self.W_1)
            self.W_1 = self.W_1 - self.critic_lr * self.evn.T * m @ (sigma.T @ self.W_1 + utility)

            print('step-' + str(step) + ': utility:' + str(utility.data.numpy().tolist()) + '; critic_value:' + str(
                critic_value.data.numpy().tolist()) + '\n u:' + str(u.data.numpy().tolist()))
