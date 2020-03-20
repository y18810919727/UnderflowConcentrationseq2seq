#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
from tqdm import tqdm

import torch
import time
from matplotlib import pyplot as plt
from config import args as config
from custom_dataset import Target_Col
from tensorboardX import SummaryWriter


def test_net(net, test_loader, use_cuda, loss_func=None, epoch=0, plt_visualize=False, tb_visualize=False):

    if loss_func is None:
        loss_func = torch.nn.MSELoss()
    net.eval()
    total_test_loss = 0
    total_test_items = 0

    use_cuda = use_cuda if torch.cuda.is_available() else False
    if use_cuda:
        net = net.cuda()

    for i, data in enumerate(test_loader):
        pre_x, pre_y, forward_x, forward_y = data
        pre_x = pre_x.permute(1,0,2)
        pre_y = pre_y.permute(1,0,2)
        forward_x = forward_x.permute(1,0,2)
        forward_y = forward_y.permute(1,0,2)
        if use_cuda:
            pre_x = pre_x.cuda()
            pre_y = pre_y.cuda()
            forward_x = forward_x.cuda()
            forward_y = forward_y.cuda()

        y_estimate_all = net((pre_x, pre_y, forward_x))
        test_loss = loss_func(y_estimate_all, forward_y)
        total_test_loss += float(test_loss) * pre_x.shape[1]
        total_test_items += pre_x.shape[1]

        y_est_series = y_estimate_all.detach().cpu().numpy()
        y_series = forward_y.detach().cpu().numpy()
        pre_y_cpu = pre_y.detach().cpu().numpy()

        from custom_dataset import Target_Col
        if len(Target_Col) == 3:
            name = ['height', 'UC', 'Pressure']
        else:
            name = ['UC', 'Pressure']
        if tb_visualize:
            for y_index in range(len(Target_Col)):
                for series_index in range(min(config.batch_size, pre_x.shape[1])):

                    fig = plt.figure(figsize=(5, 3))
                    #plt.ylim(-3,3)

                    plt.plot(np.arange(0, config.look_back, 1), pre_y_cpu[:, series_index, y_index])
                    right_end = y_series.shape[0] + config.look_back
                    plt.plot(np.arange(config.look_back, right_end,1),
                             y_series[:, series_index, y_index])
                    plt.plot(np.arange(config.look_back, right_end,1),
                             y_est_series[:, series_index, y_index])
                    plt.legend(['Pre', 'Real', 'forecast'])
                    plt.title('{}-{}-{}'.format(name[y_index],i,series_index))
                    # writer.add_pr_curve(tag=os.path.join(name[y_index], str(series_index)),
                    #                     predictions=y_est_series[:, series_index, y_index],
                    #                     labels=y_series[:, series_index, y_index],
                    #                     global_step=epoch,
                    #                     )
                    config.writer.add_figure(os.path.join(name[y_index], str(i), str(series_index)), fig, global_step=epoch)
                    fig.clf()


        if plt_visualize and pre_x.shape[1]>=3:
            plt.figure(figsize=(10, 6))
            for y_index in range(len(Target_Col)):
                for series_index in range(min(config.batch_size, 3)):
                    plt.subplot('33'+str(series_index+y_index*3+1))
                    plt.plot(y_series[:, series_index, y_index])
                    plt.plot(y_est_series[:, series_index, y_index])
                    plt.legend(['Real', 'forecast'])
                    plt.title('{}-{}-{}'.format(name[y_index],i,series_index))

            plt.show()

    total_test_loss /= total_test_items
    return total_test_loss, total_test_items

def train_net(net, train_loader, test_loader, config):

    train_loss_list = []
    test_loss_list = []

    use_cuda = config.use_cuda if torch.cuda.is_available() else False
    if use_cuda:
        net = net.cuda()

    optim = torch.optim.Adam(net.parameters())
    schedualer = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.9, last_epoch=-1)

    if config.loss_func == 'L2':
        critic = torch.nn.MSELoss()
    else:
        critic = torch.nn.MSELoss()

    writer = config.writer

    best_loss = 1e8
    best_info = (-1,1e8)
    bar = tqdm(total=config.epochs*len(train_loader))

    for epoch in range(0, config.epochs):

        net.train()

        start_time = time.time()
        total_train_loss = 0
        total_train_items = 0
        for i, data in enumerate(train_loader):
            bar.update(1)
            pre_x, pre_y, forward_x, forward_y = data

            pre_x = pre_x.permute(1,0,2)
            pre_y = pre_y.permute(1,0,2)
            forward_x = forward_x.permute(1,0,2)
            forward_y = forward_y.permute(1,0,2)
            if use_cuda:
                pre_x = pre_x.cuda()
                pre_y = pre_y.cuda()
                forward_x = forward_x.cuda()
                forward_y = forward_y.cuda()

            prepare_time = time.time()-start_time
            y_estimate_all = net((pre_x, pre_y, forward_x))
            loss = critic(y_estimate_all, forward_y)
            total_train_loss += float(loss)*pre_x.shape[1]
            total_train_items += pre_x.shape[1]
            optim.zero_grad()
            if 'ode' in config.algorithm:
                net.ode_net.cum_t = 0
                time_beg = time.time()
                loss.backward()
                print('backward %i %f s' % (net.ode_net.cum_t, time.time() - time_beg))
            else:
                loss.backward()



            optim.step()

            # write to tensorboard

            process_time = time.time() - start_time - prepare_time
            #pbar.set_description("compute efficiency: {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time), epoch, config.epochs))
            #print("compute efficiency: {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time), epoch, config.epochs))
            start_time = time.time()


        writer.add_scalar('train_loss', total_train_loss/total_train_items, epoch)
        train_loss_list.append(total_train_loss/total_train_items)
        #print('train loss after {} epoch is {:.2f}'.format(epoch+1, total_train_loss/total_train_items))
        schedualer.step()

        if epoch % config.test_period == config.test_period - 1:

            total_test_loss, _ = test_net(net, test_loader, use_cuda, critic, epoch=epoch,
                                          plt_visualize=config.plt_visualize, tb_visualize=config.tb_visualize)

            #print('test loss after {} epoch is {:.2f}'.format(epoch+1, total_test_loss))
            writer.add_scalar('test_loss', total_test_loss, epoch)
            test_loss_list.append(total_test_loss)

            if not os.path.exists('./ckpt/'+config.save_dir):
                os.makedirs('./ckpt/'+config.save_dir)
            if total_test_loss < best_loss:
                best_loss = min(total_test_loss, best_loss)
                best_info = (epoch, total_train_loss/total_train_items)
                state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'optim': optim.state_dict(),
                    'scaler_mean': config.scaler.mean_,
                    'scaler_var': config.scaler.var_
                }
                torch.save(state, os.path.join('ckpt', config.save_dir, str(epoch))+'.pth')

    print('best loss = {:.4f} in epoch = {} with train_loss = {:.4f}'.format(best_loss, best_info[0], best_info[1]))
    return net, train_loss_list, test_loss_list

