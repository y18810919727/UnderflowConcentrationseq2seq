#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
import numpy as np
import math
import os
import cv2 as cv
import json
import torch
import config
from tqdm import tqdm
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from matplotlib import pyplot as plt
import traceback


from train_and_test import test_net, train_net
from common import cal_params_sum
from common import RRSE
from models.model_generator import initialize_model
from custom_dataset import initialize_dataset


def set_random_seed(config):

    if config.random_seed is None:
        rand_seed = np.random.randint(0,100000)
    else:
        rand_seed = config.random_seed
    print('random seed = {}'.format(rand_seed))
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


def main(config):

    set_random_seed(config)
    net = initialize_model(config)
    data, scaled_data, train_loader, val_loader, test_loader = initialize_dataset(config)
    cal_params_sum(net)
    if config.is_train:
        from common import SimpleLogger
        logging = SimpleLogger(os.path.join('ckpt', config.save_dir, 'log.out'))
        logging('save_dir: %s' % config.save_dir)
        try:
            net, train_loss_list, val_loss_list = train_net(net, train_loader, val_loader, test_loader, logging, config)
        except Exception as e:
            var = traceback.format_exc()
            logging(var)

    # elif config.test_all:
    #
    #     state = torch.load(os.path.join('ckpt', config.save_dir, str(config.test_model)+'.pth'))
    #     net.load_state_dict(state['net'])
    #     all_data_lenghth = len(data)
    #     one_sequence_dataset = MyDataset(pd.DataFrame(scaled_data, columns=data.columns, index=data.index),
    #                                      look_back=config.look_back, look_forward=config.look_forward,
    #                                      sample_dis=config.look_back+config.look_forward)
    #     test_loader = DataLoader(one_sequence_dataset, batch_size=1, shuffle=False)
    #     # import pdb
    #     # pdb.set_trace()
    #     print(test_net(net, test_loader, config.use_cuda,  plt_visualize=False, tb_visualize=config.tb))
    else:
        state = torch.load(os.path.join('ckpt', config.save_dir, str(config.test_model)+'.pth'))
        net.load_state_dict(state['net'])
        print(config.save_dir, test_net(net, val_loader, config.use_cuda,
                                        critic_func={'RRSE': RRSE, 'MSE': torch.nn.MSELoss()},
                                        plt_visualize=False, tb_visualize=False))

    if 'writer' in config.__dict__.keys():
        config.writer.close()


if __name__ == '__main__':

    from config import args as config
    main(config)



