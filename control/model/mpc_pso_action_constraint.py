import numpy as np
import random
import torch
from config import args


def pos_linear_tran(x):
    y = x.mul(torch.FloatTensor([1.62, 1.82, 2.52])) + torch.FloatTensor([-0.14, -1.16, 1.26 ])
    return y


def vel_linear_tran(x):
    y = x.mul(torch.FloatTensor([1, 1.12, 1.55]))
    return y

class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim):
        self.__pos = pos_linear_tran(torch.FloatTensor([random.uniform(-x_max, x_max) for i in range(dim)]))  # 粒子的位置
        self.__vel = vel_linear_tran(torch.FloatTensor([random.uniform(-max_vel, max_vel) for i in range(dim)])) # 粒子的速度
        self.__bestPos = torch.FloatTensor([0.0 for i in range(dim)])  # 粒子最好的位置
        # self.__fitnessValue = fit_fun(self.__pos)  # 适应度函数值
    def set_pos(self, value):
        self.__pos = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, value):
        self.__bestPos = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, value):
        self.__vel = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class PSO:
    def __init__(self, dim, size, iter_num, x_max, max_vel, model_pre, best_fitness_value=float('Inf'), C1=2, C2=2, W=1, error_limit=1e-3):
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.model_pre = model_pre
        self.best_fitness_value = best_fitness_value
        self.best_position = torch.FloatTensor([1.6, 1.8, 2.5])  # 种群最优位置
        # self.fitness_val_list = []  # 每次迭代最优适应值
        self.error_limit = error_limit

        self.x = self.model_pre.x
        self.y_target = self.model_pre.scaler.scale_target(torch.FloatTensor(args.y_target))
        self.Q = torch.diag(torch.FloatTensor([10.0, 0.001]))
        self.R = torch.diag(torch.FloatTensor([0.001, 0.001, 0.001]))

        self.u_range = [(-1.76, 1.48), (-2.98, 0.66), (-1.39, 3.78)]
        self.max_vel_range = [self.max_vel, self.max_vel*1.12, self.max_vel*1.55]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, value):
        self.best_position = value

    def get_bestPosition(self):
        return self.best_position

    def fit_fun(self, u):
        u = torch.unsqueeze(u, dim=0)
        du = self.last_u - u
        y, dx_dt = self.model_pre.f_pre(self.x, u)
        y_det = y - self.y_target
        y_cost = torch.sum(y_det @ self.Q @ y_det.T, dim=1)
        u_cost = torch.sum(du @ self.R @ du.T, dim=1)
        cost = (y_cost + u_cost).data.numpy()[0]
        return cost

    # 更新速度
    def update_vel(self, part):
        # for i in range(self.dim):
        vel_value = self.W * part.get_vel() + self.C1 * torch.rand(3).mul(part.get_best_pos() - part.get_pos()) \
                    + self.C2 * torch.rand(3).mul(self.get_bestPosition() - part.get_pos())
        for i in range(self.dim):
            if vel_value[i] > self.max_vel_range[i]:
                vel_value[i] = self.max_vel_range[i]
            elif vel_value[i] < -self.max_vel_range[i]:
                vel_value[i] = -self.max_vel_range[i]
        part.set_vel(vel_value)

    # 更新位置
    def update_pos(self, part):
        # for i in range(self.dim):
        pos_value = part.get_pos() + part.get_vel()
        part.set_pos(pos_value)
        value = self.fit_fun(part.get_pos())
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            part.set_best_pos(part.get_pos())
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            self.set_bestPosition(part.get_pos())

    def update(self):
        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim) for i in range(self.size)]
        for part in self.Particle_list:
            part.set_fitness_value(self.fit_fun(part.get_pos()))
        self.best_fitness_value = 100
        # last_fitness_value = 100
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            # self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
            # if abs(self.get_bestFitnessValue() - last_fitness_value) < self.error_limit:
            #     break
            # last_fitness_value = self.get_bestFitnessValue()
            # if i == 10:
            #     print('cost10=='+str(self.get_bestFitnessValue()))
        return self.get_bestFitnessValue(), self.get_bestPosition()