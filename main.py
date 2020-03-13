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

import torch
from config import args as config
from common import cal_params_sum
from custom_dataset import MyDataset, Target_Col, Control_Col
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

import pandas as pd
from train_and_test import test_net, train_net


from matplotlib import pyplot as plt
DATA_PATH = './data/res_all_selected_features_0.csv'

if config.random_seed is None:
    rand_seed = np.random.randint(0,100000)
else:
    rand_seed = config.random_seed

if config.algorithm == 'diff':
    from models.diff import DiffNet as SeriesNet
elif config.algorithm == 'ode':
    from models.ode import MyODE as SeriesNet
elif config.algorithm == 'ode_affine':
    from models.ode_affine_linear import MyODEAffine as SeriesNet

print('random seed = {}'.format(rand_seed))
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)



data = pd.read_csv(DATA_PATH)
scaler = StandardScaler().fit(data)
scaled_data = scaler.transform(data)
dataset = MyDataset(pd.DataFrame(scaled_data, columns=data.columns, index=data.index),
                    look_back=config.look_back, look_forward=config.look_forward,
                    sample_dis=config.look_back+config.look_forward)
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=[train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
print(config)

net = SeriesNet(input_size=len(Target_Col+Control_Col),
                num_layers=config.num_layers, hidden_size=config.hidden_num, out_size=len(Target_Col), net_type=config.net_type)

cal_params_sum(net)

if config.is_train:
    net, train_loss_list, test_loss_list = train_net(net, train_loader, test_loader, config)
    plt.plot(np.arange(1,config.epochs+1,1), train_loss_list)
    plt.plot(np.arange(config.test_period, config.epochs+1, config.test_period), test_loss_list)
    plt.legend(['Train loss', 'Test loss'])
    plt.show()
elif config.test_all:

    state = torch.load(os.path.join('ckpt', config.save_dir, str(config.test_model)))
    net.load_state_dict(state['net'])
    all_data_lenghth = len(data)
    one_sequence_dataset = MyDataset(pd.DataFrame(scaled_data, columns=data.columns, index=data.index),
                                     look_back=config.look_back, look_forward=config.look_forward,
                                     sample_dis=config.look_back+config.look_forward)
    test_loader = DataLoader(one_sequence_dataset, batch_size=1, shuffle=False)
    # import pdb
    # pdb.set_trace()
    print(test_net(net, test_loader, config.use_cuda,  plt_visualize=False, tb_visualize=config.tb_visualize))

else:
    state = torch.load(os.path.join('ckpt', config.save_dir, str(config.test_model)))
    net.load_state_dict(state['net'])
    print(test_net(net, test_loader, config.use_cuda,  plt_visualize=False, tb_visualize=False))

if 'writer' in config.__dict__.keys():
    config.writer.close()

