import numpy as np
import math
import os
import torch
import time
from control.thickener import Thickener
from custom_dataset import Control_Col, Target_Col
from control.scaler import MyScaler
import common
from control.model.mpc_pso import PSO
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
            self.writer = SummaryWriter(os.path.join('logs', 'mpc_control', str(
                time.strftime("%Y%m%d%H%M%S", time.localtime())) + '__' + parameter_name + '__T=' + str(self.evn.T))+ '__W='+str(self.pso.W)\
                                        + '__C1=='+str(self.pso.C1) + '__C2==' + str(self.pso.C2) + '__size=' + str(self.pso.size))

        print(parameter_name)

    def start(self):
        step = 0
        # pi = 3.14159265359
        while(True):
            bestFitnessValue, du = self.pso.update()
            u = self.last_u + torch.unsqueeze(torch.FloatTensor(du),dim=0)
            self.last_u = u
            x, dxdt = self.evn.f(u)
            y = self.evn.scaler.unscale_target(x)
            print('step'+str(step)+':'+'state:' + str(y.data.numpy().tolist()) + '\ncost:' + str(bestFitnessValue))

            if config.is_write > 0:
                self.writer.add_scalar('Cost', bestFitnessValue, step)
                if config.action_constraint > 0:
                    u_unscale = self.evn.scaler.unscale_controllable(torch.atan(u)*2/pi)
                else:
                    u_unscale = self.evn.scaler.unscale_controllable(u)

                for i in range(len(self.evn.controllable_in_input_index)):
                    self.writer.add_scalar('Actor/u' + str(i), u_unscale.data[0][i], step)

                self.writer.add_scalar('State/Concentration of underﬂow', y.data[0][0], step)
                self.writer.add_scalar('State/Height of mud layer', y.data[0][1], step)

                c_unscale = self.evn.scaler.unscale_uncontrollable(self.evn.c)
                self.writer.add_scalar('Noise/noise1', c_unscale.data[0][0], step)
                self.writer.add_scalar('Noise/noise2', c_unscale.data[0][1], step)

                self.writer.add_scalar('dxdt', dxdt.data[0][0], step)
            step = step + 1

            self.pso.last_u = u
            self.pso.model_pre.t = self.evn.t
            self.pso.x = self.evn.x

            if step == 301:
                break

def load_model(state_dic, random_seed=None):
    # 保存的pth文件中直接记录着当时训练模型时的config字典
    model_config = state_dic['config']

    net = initialize_model(config=model_config)

    net.load_state_dict(state_dic['net'])
    net.ode_net.interpolation_kind = 'slinear'

    _mean, _var = state_dic['scaler_mean'], state_dic['scaler_var']
    my_scaler = MyScaler(_mean, _var, model_config.all_col, Target_Col, Control_Col, config.controllable,
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
    random_seed = None
    thickener_controlled = load_model(state_dic_controlled, random_seed)
    thickener_prediction = load_model(state_dic_prediction, thickener_controlled.random_seed)

    # last_u = torch.FloatTensor([0, 0, 0])
    # y_target = scaler.scale_target(torch.FloatTensor(args.y_target))

    pso = PSO(dim=3, size=40, iter_num=20, model_pre=thickener_prediction, x_max=0.6, max_vel=0.3, best_fitness_value=100)
    # pso = PSO(dim=3, size=50, iter_num=10, evn=thickener, x_max=10, max_vel=5, best_fitness_value=100)

    mpc_control = MPC_Control(pso=pso, evn=thickener_controlled)

    mpc_control.start()



