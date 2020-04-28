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
from datetime import datetime

class SynchronousController:

    def __init__(self,
                 evn: Thickener,
                 quadratic_cost: QuadraticCost,
                 scaler: MyScaler,
                 gpu_id=0,
                 iter_rounds=500,
                 actor_err_limit=1e-4,
                 critic_err_limit=0.01,
                 actor_lr=1,
                 critic_lr=1,
                 dim_x_c=16,
                 step_max=200000
                 ):

        self.evn = evn
        self.quadratic_cost = quadratic_cost
        self.scaler = scaler
        self.gpu_id = gpu_id
        self.iter_rounds = iter_rounds
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_err_limit = actor_err_limit
        self.critic_err_limit = critic_err_limit
        self.step_max = step_max

        if config.x_decode > 0:
            self.dim_x_c = 2
        else:
            self.dim_x_c = dim_x_c

        # self.W_size = int(self.dim_x_c * (self.dim_x_c+1) / 2)
        self.W_size = self.dim_x_c ** 2
        self.W_1 = torch.ones(self.W_size, 1) / 2
        self.W_2 = torch.ones(self.W_size, 1) / 2

        f_value = self.W_size * [1]

        self.F_1 = torch.unsqueeze(torch.FloatTensor(f_value), dim=1)
        self.F_2 = torch.diag(torch.FloatTensor(f_value))

        parameter_name = 'y_target=' + str(config.y_target)
        if config.is_write > 0:
            self.writer = SummaryWriter(os.path.join('logs', 'ode_control', str(
                time.strftime("%Y%m%d%H%M%S", time.localtime())) + '__' + parameter_name))
        print(parameter_name)

    def phi(self, x):
        """
        多项式向量
        :param x: 输入
        :return:
        """
        assert x.shape[1] == self.dim_x_c
        x = Variable(x.data, requires_grad=True)
        # phi_value = torch.squeeze(torch.cat((x*x, x), dim=1))
        phi_value = x.T @ x
        phi_value = phi_value.view(self.W_size)
        jacT = torch.zeros(self.dim_x_c, self.W_size)
        # start_time = datetime.now()
        for i in range(self.W_size):
            output = torch.zeros(self.W_size)
            output[i] = 1
            j = torch.autograd.grad(phi_value, x, grad_outputs=output, retain_graph=True)
            jacT[:, i:i + 1] = j[0].T
        # end_time = datetime.now()
        # print('time_phi:' + str(end_time-start_time))
        return phi_value.data, jacT.data

    def dh_du_func(self, u, w, c):
        """
        计算 f对u的导数 * w
        :param u:
        :param w: phi @ W_2
        :return: f_u_grad
        """
        u = Variable(u, requires_grad=True)
        grad_module = self.evn.ode_net.grad_module

        ode_input = torch.cat((self.evn.last_x, c[:, 0:1], u, c[:, 1:2]), dim=1)
        ode_ouput = grad_module(ode_input)
        if config.x_decode > 0:
            ode_ouput = self.evn.fcn(ode_ouput)
        j = torch.autograd.grad(ode_ouput, u, grad_outputs=w.T, create_graph=True)

        f_u_grad = j[0]

        h_u = u + f_u_grad @ self.quadratic_cost.R.inverse() / 2

        h_u_grad = torch.zeros(h_u.shape[1], u.shape[1])
        for i in range(h_u.shape[1]):
            gradients = torch.zeros(1, h_u.shape[1])
            gradients[:, i] = 1
            if i == h_u.shape[1] - 1:
                j = torch.autograd.grad(h_u[:, i], u)
            else:
                j = torch.autograd.grad(h_u[:, i], u, retain_graph=True)
            h_u_grad[:, i:i + 1] = j[0].T

        return h_u.data @ h_u_grad.inverse().data

    def solve_u(self, w, last_u):
        u = last_u
        i = 0
        c = self.evn.c
        lr = 1
        while True:
            i += 1
            if lr % 10 == 0:
                lr = lr * 0.8
            adjust_value = self.dh_du_func(u, w, c)
            new_u = u - adjust_value.data * lr
            if torch.dist(new_u, u) < self.actor_err_limit:
                break
            last_dist = torch.dist(new_u, u)
            # if i == 10:
            print('u_dist:' + str(last_dist.data.numpy().tolist()))
            u = new_u.data
        return new_u

    def train(self):

        torch.cuda.set_device(self.gpu_id)
        # u = torch.rand(1, 3)
        u = torch.FloatTensor([[0, 0, 0]])

        for step in range(self.step_max):
            print('STEP-'+str(step))
            x, x_grad, f_u_grad = self.evn.f(u)

            utility = self.quadratic_cost.get_cost(self.evn.last_x, u)
            # phi, phi_x_grad = self.phi(torch.cat((x, self.evn.c), dim=1))
            phi, phi_x_grad = self.phi(x)
            critic_value = self.W_1.T @ phi


            u = self.solve_u(last_u=u, w=phi_x_grad.data @ self.W_2.data)
            # u = (- self.quadratic_cost.R.inverse() @ f_u_grad @ phi_x_grad @ self.W_2 / 2).T

            u_unscale = self.scaler.unscale_controllable(u)
            if config.is_write > 0:
                self.writer.add_scalar('Critic/Reward', utility.data[0], step)
                self.writer.add_scalar('Critic/Value', critic_value.data[0], step)

                for i in range(len(self.evn.controllable_in_input_index)):
                    self.writer.add_scalar('Actor/u' + str(i), u_unscale.data[0][i], step)

                if config.x_decode > 0:
                    y = x
                else:
                    y = self.quadratic_cost.fcn(x)
                y = self.scaler.unscale_target(y)
                self.writer.add_scalar('State/Concentration of underﬂow', y.data[0][0], step)
                self.writer.add_scalar('State/Height of mud layer', y.data[0][1], step)

                c_unscale = self.scaler.unscale_uncontrollable(self.evn.c)
                self.writer.add_scalar('Noise/noise1', c_unscale.data[0][0], step)
                self.writer.add_scalar('Noise/noise2', c_unscale.data[0][1], step)

            # 参数更新
            sigma = phi_x_grad.T @ x_grad.T
            sigma_hat = sigma / (sigma.T @ sigma + 1)
            m = sigma / (sigma.T @ sigma + 1) ** 2
            D1 = phi_x_grad.T @ f_u_grad.T @ self.quadratic_cost.config['R'].inverse() @ f_u_grad @ phi_x_grad

            self.W_2 = self.W_2 - self.actor_lr * self.evn.T * (
                    self.F_2 @ self.W_2 - self.F_1 @ sigma_hat.T @ self.W_1 - D1 @ self.W_2 @ m.T @ self.W_1)
            self.W_1 = self.W_1 - self.critic_lr * self.evn.T * m @ (sigma.T @ self.W_1 + utility)

            self.F_1 = - D1 @ self.W_1 / (sigma.T @ sigma + 1) / 4
            self.F_2 = (D1 @ self.W_1 @ m.T + m @ self.W_1.T @ D1) / 8 + torch.diag(torch.FloatTensor(self.W_size * [1]))

            print('utility:' + str(utility.data.numpy().tolist()) + '; critic_value:' + str(
                critic_value.data.numpy().tolist()) + '\n u:' + str(u_unscale.data.numpy().tolist()))
