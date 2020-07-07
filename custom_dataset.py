#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch.utils.data.dataset import Dataset

from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd

class MyDataset(Dataset):
    def __init__(self,
                 data,
                 config,
                 look_back=120,
                 look_forward=60,
                 sample_dis=-1,
                 ):
        """
        :param data: 被normalize之后的数据
        :param look_back:
        :param look_forward:
        """
        self.data_length = len(data)
        if sample_dis == -1:
            sample_dis = look_back + look_forward

        if look_forward == -1 or look_forward > self.data_length - look_back:
            look_forward = self.data_length - look_back
            sample_dis = self.data_length

        self.config = config
        self.sample_dis = sample_dis
        self.look_forward = look_forward
        self.look_back = look_back
        self.data = data
        self.Target_Col = config.Target_Col
        self.Control_Col = config.Control_Col


        #     pre_x = data[ind:ind+look_back][Control_Col]
        #     pre_y = data[ind:ind+look_back][Target_Col]
        #     forward_x = data[ind+look_back:ind+look_back+look_forward][Control_Col]
        #     forward_y = data[ind+look_back:ind+look_back+look_forward][Target_Col]
        #
        #     self.pre_X_all.append(pre_x)
        #     self.pre_Y_all.append(pre_y)
        #     self.forward_X_all.append(forward_x)
        #     self.forward_Y_all.append(forward_y)
        #
        # self.pre_X_all = np.stack(self.pre_X_all, axis=0)
        # self.pre_Y_all = np.stack(self.pre_Y_all, axis=0)
        # self.forward_X_all = np.stack(self.forward_X_all, axis=0)
        # self.forward_Y_all = np.stack(self.forward_Y_all, axis=0)


    def __getitem__(self, item):

        ind = item * self.sample_dis
        data = self.data
        look_back = self.look_back
        look_forward = self.look_forward

        pre_x = data[ind:ind+look_back][self.Control_Col]
        pre_y = data[ind:ind+look_back][self.Target_Col]
        forward_x = data[ind+look_back:ind+look_back+look_forward][self.Control_Col]
        forward_y = data[ind+look_back:ind+look_back+look_forward][self.Target_Col]

        tmp = [np.asarray(x).astype(np.float32) for x in [
            pre_x,
            pre_y,
            forward_x,
            forward_y,
        ]]

        # Autoregression vs System Identification
        if self.config.nou:
            tmp[2] = tmp[2] * 0
        return tuple(tmp)


    def __len__(self):
        return int((self.data_length - self.look_back - self.look_forward + self.sample_dis ) / self.sample_dis)



def initialize_dataset(config):

    scaler = config.scaler
    data = pd.read_csv(config.DATA_PATH)
    scaled_data = scaler.transform(data)
    [train_size, validate_size, test_size] = [int(len(scaled_data)*r) for r in [0.7, 0.15, 0.15]]

    if config.data_inv:
        half_train_size = int(train_size/2)
        scaled_data_train = np.concatenate([
            scaled_data[0:half_train_size], scaled_data[-(train_size-half_train_size):]
        ], axis=0)

        scaled_data_validate = scaled_data[half_train_size:validate_size+half_train_size]
        scaled_data_test = scaled_data[half_train_size + validate_size: half_train_size + validate_size + test_size]

    else:
        scaled_data_train = scaled_data[:train_size]
        scaled_data_validate = scaled_data[train_size:validate_size+train_size]
        scaled_data_test = scaled_data[-test_size:]


    train_dataset = MyDataset(pd.DataFrame(scaled_data_train, columns=data.columns), config,
                              look_back=config.look_back, look_forward=config.look_forward,
                              sample_dis=config.sample_dis)

    validate_dataset = [MyDataset(pd.DataFrame(scaled_data_validate, columns=data.columns), config,
                                 look_back=config.look_back, look_forward=look_forward) for look_forward in config.test_look_forward]

    test_dataset = [MyDataset(pd.DataFrame(scaled_data_test, columns=data.columns), config,
                             look_back=config.look_back, look_forward=look_forward) for look_forward in config.test_look_forward]


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=16)
    val_loader = [DataLoader(x, batch_size=config.batch_size, shuffle=False) for x in validate_dataset]
    test_loader = [DataLoader(x, batch_size=config.batch_size, shuffle=False) for x in test_dataset]
    return data, scaled_data, train_loader, val_loader, test_loader

