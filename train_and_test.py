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
import copy
from custom_dataset import Target_Col
from tensorboardX import SummaryWriter
from common import RRSE


def test_net(net, test_loader, config, writer, critic_func=None, epoch=0, plt_visualize=False, tb_visualize=False):

    if critic_func is None:
        critic_func = torch.nn.MSELoss()
    total_test_items, acc_time = 0, 0

    if type(critic_func) is dict and len(critic_func) >=2:
        metrics = {}
        for key, func in critic_func.items():
            metric, total_test_items, acc_time = test_net(net, test_loader, config, writer, func, epoch, plt_visualize, tb_visualize)
            metrics[key] = metric
        return metrics, total_test_items, acc_time

    net.eval()
    total_test_loss = 0
    total_test_items = 0

    use_cuda = config.use_cuda if torch.cuda.is_available() else False
    if use_cuda:
        net = net.cuda()

    acc_time = 0
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

        cur_time = time.time()
        y_estimate_all = net((pre_x, pre_y, forward_x))
        used_time = time.time() - cur_time
        acc_time += used_time

        test_loss = critic_func(y_estimate_all, forward_y)
        total_test_loss += float(test_loss) * pre_x.shape[1]
        total_test_items += pre_x.shape[1]

        y_est_series = y_estimate_all.detach().cpu()
        y_series = forward_y.detach().cpu()
        pre_y_cpu = pre_y.detach().cpu()

        # unscale series by self defined scaler
        pre_y_cpu = config.my_scaler.unscale_target(pre_y_cpu).cpu()
        y_series = config.my_scaler.unscale_target(y_series).cpu()
        y_est_series = config.my_scaler.unscale_target(y_est_series).cpu()

        from custom_dataset import Target_Col
        if len(Target_Col) == 3:
            name = ['height', 'UC', 'Pressure']
            unit_name = ['m', '%', 'Mpa']
        else:
            name = ['UC', 'Pressure']
            unit_name = ['%', 'Mpa']
        if tb_visualize:
            for y_index in range(len(Target_Col)):
                for series_index in range(min(config.batch_size, pre_x.shape[1])):

                    # ODE method performs best in group : (1-0)
                    if i!=1 or series_index != 0:
                        continue

                    fig = plt.figure(figsize=(5, 4))
                    #plt.ylim(-3,3)

                    plt.plot(np.arange(0, config.look_back, 1), pre_y_cpu[:, series_index, y_index])
                    right_end = y_series.shape[0] + config.look_back
                    plt.plot(np.arange(config.look_back, right_end,1),
                             y_series[:, series_index, y_index])
                    plt.plot(np.arange(config.look_back, right_end,1),
                             y_est_series[:, series_index, y_index])
                    plt.legend(['History', 'True', 'Prediction'])
                    plt.grid()
                    if config.is_train == '0':
                        plt.title('{}-{}-{}'.format(name[y_index],i,series_index))
                    else:
                        plt.ylabel(name[y_index]+'({})'.format(unit_name[y_index]))
                        plt.xlabel('Time(min)')
                    # writer.add_pr_curve(tag=os.path.join(name[y_index], str(series_index)),
                    #                     predictions=y_est_series[:, series_index, y_index],
                    #                     labels=y_series[:, series_index, y_index],
                    #                     global_step=epoch,
                    #                     )
                    #plt.savefig('expresults/figs/'+ name[y_index] + '_' +config.save_dir+'.eps', dpi=600)
                    writer.add_figure(os.path.join(name[y_index], str(i), str(series_index)), fig, global_step=epoch)
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
    return total_test_loss, total_test_items, acc_time

def train_net(net, train_loader, val_loader, config):

    train_loss_list = []
    val_RRSE_list = []

    use_cuda = config.use_cuda if torch.cuda.is_available() else False
    if use_cuda:
        net = net.cuda()


    if config.loss_func == 'L2':
        critic = torch.nn.MSELoss()
    elif config.loss_func == 'gauss':
        from common import GaussLoss
        critic = GaussLoss(len(Target_Col), config)
        sigma_optim = torch.optim.Adam([critic.sigma])


    optim = torch.optim.Adam(net.parameters(), lr=5e-4)
    schedualer = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.95, last_epoch=-1)

    from common import MyWriter

    writer = MyWriter(
        save_path=config.tb_path,
        is_write=config.tb

    )

    best_RRSE = 1e8
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

            if config.loss_func == 'gauss':
                sigma_optim.zero_grad()


            if 'ode' in config.algorithm:
                net.ode_net.cum_t = 0
                time_beg = time.time()
                loss.backward()
                #print('backward %i %f s' % (net.ode_net.cum_t, time.time() - time_beg))
            else:
                loss.backward()

            optim.step()
            if config.loss_func == 'gauss':
                sigma_optim.step()

            # write to tensorboard

            process_time = time.time() - start_time - prepare_time
            #pbar.set_description("compute efficiency: {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time), epoch, config.epochs))
            #print("compute efficiency: {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time), epoch, config.epochs))
            start_time = time.time()


        writer.add_scalar('train_loss', total_train_loss/total_train_items, epoch)
        if config.loss_func =='gauss':
            for i in range(critic.sigma.shape[0]):
                writer.add_scalar('sigma'+str(i), float(critic.get_cov()[i,i].cpu()), epoch)
        train_loss_list.append(total_train_loss/total_train_items)
        #print('train loss after {} epoch is {:.2f}'.format(epoch+1, total_train_loss/total_train_items))
        schedualer.step()

        if epoch % config.test_period == config.test_period - 1:

            total_val_metrics, _, acc_time = test_net(net, val_loader, config, writer, {'RRSE': RRSE, 'MSE': torch.nn.MSELoss()}, epoch=epoch,
                                          plt_visualize=config.plt_visualize, tb_visualize=config.tb)

            total_val_RRSE = total_val_metrics['RRSE']
            total_val_MSE = total_val_metrics['MSE']
            #print('val loss after {} epoch is {:.2f}'.format(epoch+1, total_val_loss))

            writer.add_scalar('val_RRSE', total_val_RRSE, epoch)
            writer.add_scalar('val_MSE', total_val_MSE, epoch)
            writer.add_scalar('val_time', acc_time, epoch)
            val_RRSE_list.append(total_val_RRSE)

            if total_val_RRSE < best_RRSE:
                best_RRSE = min(total_val_RRSE, best_RRSE)
                best_info = (epoch, total_train_loss/total_train_items)
                state = {
                    'net':  copy.deepcopy(net.state_dict()),
                    'epoch': epoch,
                    'optim': optim.state_dict(),
                    'scaler_mean': config.scaler.mean_,
                    'scaler_var': config.scaler.var_,
                    'config': copy.deepcopy(config)
                }

                # just save the best model parameters
                best_state =state
                # torch.save(state, os.path.join('ckpt', config.save_dir, str(epoch))+'.pth')

    if not os.path.exists('./ckpt/'+config.save_dir):
        os.makedirs('./ckpt/'+config.save_dir)
    torch.save(best_state, os.path.join('ckpt', config.save_dir, 'best')+'.pth')
    print('best loss = {:.4f} in epoch = {} with train_loss = {:.4f}'.format(best_RRSE, best_info[0], best_info[1]))
    return net, train_loss_list, val_RRSE_list

