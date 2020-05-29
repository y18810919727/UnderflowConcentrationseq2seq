#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from common import col2Index

class MyScaler:
    def __init__(self, scaler_mean, scaler_var, All_col, target_col, control_col, controllable=None, uncontrollable=None):
        scaler_var = np.sqrt(scaler_var)

        self.controllable = controllable
        self.uncontrollable = uncontrollable

        self.scaler_mean = scaler_mean
        self.scaler_var = scaler_var
        self.target_col = target_col
        self.control_col = control_col
        self.target_mean = torch.FloatTensor(scaler_mean[col2Index(All_col, target_col)])
        self.target_var = torch.FloatTensor(scaler_var[col2Index(All_col, target_col)])
        self.control_mean = torch.FloatTensor(scaler_mean[col2Index(All_col, control_col)])
        self.control_var = torch.FloatTensor(scaler_var[col2Index(All_col, control_col)])

        self.controllable_mean = self.control_mean[col2Index(control_col, self.controllable)]
        self.uncontrollable_mean = self.control_mean[col2Index(control_col, self.uncontrollable)]

        self.controllable_var = self.control_var[col2Index(control_col, self.controllable)]
        self.uncontrollable_var = self.control_var[col2Index(control_col, self.uncontrollable)]


    def unscale_all(self, x):
        return x * self.scaler_var + self.scaler_mean

    def unscale_target(self, x):
        return x * self.target_var + self.target_mean

    def unscale_control(self, u):
        return u * self.control_var + self.control_mean

    def unscale_uncontrollable(self, u):
        return u * self.uncontrollable_var + self.uncontrollable_mean

    def unscale_controllable(self, u):
        return u * self.controllable_var + self.controllable_mean

    def scale_target(self, x):
        return (x - self.target_mean) / self.target_var

    def scale_control(self, u):
        return (u - self.control_mean) / self.control_var

    def scale_controllable(self, u):
        return (u - self.controllable_mean) / self.controllable_var

    def scale_uncontrollable(self, u):
        return (u - self.uncontrollable_mean) / self.uncontrollable_var

    def scale_all(self, x):
        return (x - self.scaler_mean) / self.scaler_var

