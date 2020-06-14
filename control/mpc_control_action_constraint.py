import numpy as np
import math
import os
import torch
import time
from control.thickener import Thickener
from control.scaler import MyScaler
import common
from control.model.mpc_pso_action_constraint import PSO
from tensorboardX import SummaryWriter
from config import args as config
from models.model_generator import initialize_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MPC_Control:
    def __init__(self, pso, evn):
        self.pso = pso
        self.evn = evn
        self.last_u = torch.FloatTensor([0, 0, 0])
        self.pso.last_u = torch.FloatTensor([0, 0, 0])
        if config.constant_noise == 0:
            parameter_name = 'y_target=' + str(config.y_target) + '__noise'
        else:
            parameter_name = 'y_target=' + str(config.y_target)

        if config.is_write > 0:
            self.writer = SummaryWriter(os.path.join('logs', 'mpc_control_double_model', str(
                time.strftime("%Y%m%d%H%M%S", time.localtime())) + '__' + parameter_name + '__T=' + str(
                self.evn.T)) + '__W=' + str(self.pso.W) \
                                        + '__C1==' + str(self.pso.C1) + '__C2==' + str(self.pso.C2) + '__size=' + str(
                self.pso.size)+'_action_constraint')

        print(parameter_name)

    def start(self):
        step = 0
        y_list = []
        mae_list = []
        mse_list = []
        rmse_list = []
        danger = 0
        while (True):
            bestFitnessValue, u = self.pso.update()
            u = self.last_u + torch.unsqueeze(torch.FloatTensor(u), dim=0)
            x, dxdt = self.evn.f(u)
            y = self.evn.scaler.unscale_target(x)
            y_list.append(y.data.numpy().tolist()[0][0])
            if abs(y.data[0][0] - config.y_target[0]) > 0.15:
                danger = danger + 1
            elif abs(y.data[0][0] - config.y_target[0]) < 0.15 and danger > 0:
                danger = 0
                self.pso.size = 30
                self.pso.iter_num = 15
                self.pso.x_max = 0.3
            if danger == 20:
                self.pso.size = 50
                self.pso.iter_num = 25
                self.pso.x_max = 0.8
            self.last_u = u
            print(
                'step' + str(step) + ':' + 'state:' + str(y.data.numpy().tolist()) + '\ncost:' + str(bestFitnessValue))

            if config.is_write > 0:
                self.writer.add_scalar('Cost', bestFitnessValue, step)

                u_unscale = self.evn.scaler.unscale_controllable(u)

                self.writer.add_scalar('Actor/Flocculant', u_unscale.data[0][0], step)
                self.writer.add_scalar('Actor/Harrow', u_unscale.data[0][1], step)
                self.writer.add_scalar('Actor/Discharge flow', u_unscale.data[0][2], step)

                self.writer.add_scalar('State/Concentration of underﬂow', y.data[0][0], step)
                self.writer.add_scalar('State/Height of mud layer', y.data[0][1], step)

                c_unscale = self.evn.scaler.unscale_uncontrollable(self.evn.c)
                self.writer.add_scalar('Noise/Feed concentration', c_unscale.data[0][0], step)
                self.writer.add_scalar('Noise/Feed flow', c_unscale.data[0][1], step)

                self.writer.add_scalar('dxdt', dxdt.data[0][0], step)
            step = step + 1

            self.pso.last_u = u
            self.pso.model_pre.t = self.evn.t
            self.pso.x = self.evn.x

            if step % 100 == 0:
                mse = mean_squared_error(y_list, [config.y_target[0]] * len(y_list))
                mae = mean_absolute_error(y_list, [config.y_target[0]] * len(y_list))
                rmse = mse ** 0.5
                mse_list.append(mse)
                mae_list.append(mae)
                rmse_list.append(rmse)
                print('MSE：' + str(mse))
                print('RMSE：' + str(rmse))
                print('MAE：' + str(mae))

            if step == 300:
                print('MSE：' + str(mse_list))
                print('RMSE：' + str(rmse_list))
                print('MAE：' + str(mae_list))
                break


def load_model(state_dic, random_seed=None, data_path=None):
    # 保存的pth文件中直接记录着当时训练模型时的config字典
    model_config = state_dic['config']

    if data_path is not None:
        model_config.DATA_PATH = data_path
    net = initialize_model(config=model_config)

    net.load_state_dict(state_dic['net'])
    net.ode_net.interpolation_kind = 'slinear'

    _mean, _var = state_dic['scaler_mean'], state_dic['scaler_var']
    my_scaler = MyScaler(_mean, _var, model_config.all_col, model_config.Target_Col, model_config.Control_Col,
                         config.controllable,
                         config.uncontrollable)
    thickener = Thickener(net, my_scaler, random_seed, config=model_config)
    return thickener


if __name__ == '__main__':
    # 注意1：这里提供了三个模型，一个affine的，两个普通的，旧的模型不能用了
    # state_dic = torch.load('./ckpt/rnn_ode_affine_3_4_cubic_full/best.pth')
    # state_dic = torch.load('./ckpt/rnn_ode_affine_3_4_cubic_transform_full/best.pth')

    state_dic_prediction = torch.load('./ckpt/rnn_ode_affine_3_4_cubic_full/best.pth')
    state_dic_controlled = torch.load('./ckpt/rnn_ode_affine_3_4_cubic_full/best.pth')

    print(config)

    # random_seed = 546782
    # random_seed = 539667
    # random_seed = 82791
    random_seed = 55630

    thickener_controlled = load_model(state_dic_controlled, random_seed=random_seed)
    thickener_prediction = load_model(state_dic_prediction, random_seed=thickener_controlled.random_seed)


    pso = PSO(dim=3, size=30, iter_num=15, model_pre=thickener_prediction, x_max=1, max_vel=0.2,
              best_fitness_value=100)
    # pso = PSO(dim=3, size=50, iter_num=10, evn=thickener, x_max=10, max_vel=5, best_fitness_value=100)

    mpc_control = MPC_Control(pso=pso, evn=thickener_controlled)

    mpc_control.start()
