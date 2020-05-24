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
#state_dic = torch.load('./ckpt/lstm_ode_4_5/95.pth')

# 更新使用的仿真模型

from config import args as config
from models.model_generator import initialize_model

class MPC_Control:
    def __init__(self, pso, scaler):
        self.pso = pso
        self.scaler = scaler

        if config.constant_noise == 0:
            parameter_name = 'y_target=' + str(config.y_target) + '__noise'
        else:
            parameter_name = 'y_target=' + str(config.y_target)

        if config.is_write > 0:
            self.writer = SummaryWriter(os.path.join('logs', 'mpc_control', str(
                time.strftime("%Y%m%d%H%M%S", time.localtime())) + '__' + parameter_name + '__T=' + str(self.pso.evn.T))+ '__W='+str(self.pso.W)\
                                        + '__C1=='+str(self.pso.C1) + '__C2==' + str(self.pso.C2) + '__size=' + str(self.pso.size))

        print(parameter_name)

    def start(self):
        step = 0
        pi = 3.14159265359
        while(True):
            bestFitnessValue, du = self.pso.update()
            u = self.pso.evn.last_u + torch.unsqueeze(torch.FloatTensor(du),dim=0)
            x, dxdt = self.pso.evn.f(u, forward=True)
            y = self.scaler.unscale_target(x)
            print('step'+str(step)+':'+'state:' + str(y.data.numpy().tolist()) + '\ncost:' + str(bestFitnessValue))

            if config.is_write > 0:
                self.writer.add_scalar('Cost', bestFitnessValue, step)
                if config.action_constraint > 0:
                    u_unscale = self.scaler.unscale_controllable(torch.atan(u)*2/pi)
                else:
                    u_unscale = self.scaler.unscale_controllable(u)

                for i in range(len(self.pso.evn.controllable_in_input_index)):
                    self.writer.add_scalar('Actor/u' + str(i), u_unscale.data[0][i], step)

                self.writer.add_scalar('State/Concentration of underﬂow', y.data[0][0], step)
                self.writer.add_scalar('State/Height of mud layer', y.data[0][1], step)

                c_unscale = self.scaler.unscale_uncontrollable(self.pso.evn.c)
                self.writer.add_scalar('Noise/noise1', c_unscale.data[0][0], step)
                self.writer.add_scalar('Noise/noise2', c_unscale.data[0][1], step)

                self.writer.add_scalar('dxdt', dxdt.data[0][0], step)
            step = step + 1
            if step == 301:
                break


if __name__ == '__main__':
    #model_dir = 'rnn_ode_2_3_h32'
    model_dir = 'rnn_ode_affine_2_3_h16'
    model_name = 'best.pth'

    # 载入pth
    state_dic = torch.load(
        os.path.join('./ckpt', model_dir, model_name ))
    assert common.parser_dir(model_dir, config)
    print(config)
    net = initialize_model(config)
    net.load_state_dict(state_dic['net'])

    _mean, _var = state_dic['scaler_mean'], state_dic['scaler_var']
    my_scaler = MyScaler(_mean, _var, Target_Col, Control_Col, config.controllable, config.uncontrollable)

    thickener = Thickener(net, my_scaler, None)

    pso = PSO(dim=3, size=50, iter_num=10, evn=thickener, x_max=0.5, max_vel=0.3, best_fitness_value=100)
    # pso = PSO(dim=3, size=50, iter_num=10, evn=thickener, x_max=10, max_vel=5, best_fitness_value=100)

    mpc_control = MPC_Control(pso=pso, scaler=my_scaler)

    mpc_control.start()



