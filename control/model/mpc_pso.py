import numpy as np
import random
import torch

class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim):
        self.__pos = torch.FloatTensor([random.uniform(-x_max, x_max) for i in range(dim)])  # 粒子的位置
        self.__vel = torch.FloatTensor([random.uniform(-max_vel, max_vel) for i in range(dim)])  # 粒子的速度
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
    def __init__(self, dim, size, iter_num, x_max, max_vel, evn, best_fitness_value=float('Inf'), C1=2, C2=2, W=1, error_limit=5e-3):
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.evn = evn
        self.best_fitness_value = best_fitness_value
        self.best_position = torch.FloatTensor([0.0 for i in range(dim)])  # 种群最优位置
        # self.fitness_val_list = []  # 每次迭代最优适应值
        self.error_limit = error_limit

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, value):
        self.best_position = value

    def get_bestPosition(self):
        return self.best_position

    # 更新速度
    def update_vel(self, part):
        # for i in range(self.dim):
        vel_value = self.W * part.get_vel() + self.C1 * torch.rand(3).mul(part.get_best_pos() - part.get_pos()) \
                    + self.C2 * torch.rand(3).mul(self.get_bestPosition() - part.get_pos())
        for i in range(self.dim):
            if vel_value[i] > self.max_vel:
                vel_value[i] = self.max_vel
            elif vel_value[i] < -self.max_vel:
                vel_value[i] = -self.max_vel
        part.set_vel(vel_value)

    # 更新位置
    def update_pos(self, part):
        # for i in range(self.dim):
        pos_value = part.get_pos() + part.get_vel()
        part.set_pos(pos_value)
        value = self.evn.fit_fun(part.get_pos())
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
            part.set_fitness_value(self.evn.fit_fun(part.get_pos()))
        self.best_fitness_value = 100

        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            # self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
            if self.get_bestFitnessValue() < self.error_limit:
                break
            # if i == 10:
            #     print('cost10=='+str(self.get_bestFitnessValue()))
        return self.get_bestFitnessValue(), self.get_bestPosition()