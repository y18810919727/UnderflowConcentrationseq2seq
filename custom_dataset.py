#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
#Target_Col = ['1', '11', '17']
Target_Col = ['11', '17']
Control_Col = ['4','5','7','15','16']

class MyDataset(Dataset):
    def __init__(self,
                 data,
                 look_back=120,
                 look_forward=60,
                 sample_dis=180,
                 ):
        """
        :param data: 被normalize之后的数据
        :param look_back:
        :param look_forward:
        """
        length = len(data)
        self.pre_X_all = []
        self.pre_Y_all = []
        self.forward_X_all = []
        self.forward_Y_all = []
        for ind in range(0, length - (look_back + look_forward) +1, sample_dis):
            pre_x = data[ind:ind+look_back][Control_Col]
            pre_y = data[ind:ind+look_back][Target_Col]
            forward_x = data[ind+look_back:ind+look_back+look_forward][Control_Col]
            forward_y = data[ind+look_back:ind+look_back+look_forward][Target_Col]

            self.pre_X_all.append(pre_x)
            self.pre_Y_all.append(pre_y)
            self.forward_X_all.append(forward_x)
            self.forward_Y_all.append(forward_y)

        self.pre_X_all = np.stack(self.pre_X_all, axis=0)
        self.pre_Y_all = np.stack(self.pre_Y_all, axis=0)
        self.forward_X_all = np.stack(self.forward_X_all, axis=0)
        self.forward_Y_all = np.stack(self.forward_Y_all, axis=0)


    def __getitem__(self, item):
        tmp = [np.asarray(x).astype(np.float32) for x in [
            self.pre_X_all[item],
            self.pre_Y_all[item],
            self.forward_X_all[item],
            self.forward_Y_all[item],
        ]]
        return tuple(tmp)

    def __len__(self):
        return len(self.forward_X_all)














