#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import time
from matplotlib import pyplot as plt
import copy
from tensorboardX import SummaryWriter
from common import RRSE


def test_net(net, test_loader, config, writer, critic_func=None, epoch=0, plt_visualize=False, tb_visualize=False,
             data_loader_name=None):

    if critic_func is None:
        critic_func = torch.nn.MSELoss()
    if data_loader_name is None:
        data_loader_name = ""
    else:
        data_loader_name = data_loader_name + '-'
    total_test_items, acc_time = 0, 0
    used_time = {
        'prepare data': 0.0,
        'forward': 0.0,
        'evaluation and visualization': 0.0
    }
    if type(critic_func) is dict:
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

    for i, data in enumerate(test_loader):
        # if i!=0:
        #     continue
        time_point = time.time()
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
        used_time['prepare data'] += time.time() - time_point

        # Forward
        time_point = time.time()
        y_estimate_all = net((pre_x, pre_y, forward_x))
        used_time['forward'] += time.time() - time_point

        # Evaluation
        time_point = time.time()
        y_est_series = y_estimate_all.detach().cpu()
        y_series = forward_y.detach().cpu()
        pre_y_cpu = pre_y.detach().cpu()

        # unscale series by self defined scaler
        pre_y_cpu = config.my_scaler.unscale_target(pre_y_cpu).cpu()
        y_series = config.my_scaler.unscale_target(y_series).cpu()
        y_est_series = config.my_scaler.unscale_target(y_est_series).cpu()

        test_loss = critic_func(y_est_series, y_series)
        total_test_loss += float(test_loss) * pre_x.shape[1]
        total_test_items += pre_x.shape[1]

        if config.dataset_name == 'thickener':
            if len(config.Target_Col) == 3:
                name = ['height', 'UC', 'Pressure']
                unit_name = ['m', '%', 'Mpa']
            else:
                name = ['UC', 'Pressure']
                unit_name = ['%', 'Mpa']
        elif config.dataset_name == 'cstr':
            name = ['Concentration', 'Tenp']
            unit_name = ['%', 'C']
        if tb_visualize:
            for y_index in range(len(config.Target_Col)):
                for series_index in range(min(config.batch_size, pre_x.shape[1])):
                    # if series_index != 1:
                    #     continue


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
                    # plt.savefig('expresults/figs/'+str(y_series.shape[0])+'/' + data_loader_name+'_'+name[y_index] + '_' +config.save_dir
                    #             +'.eps', dpi=600)
                    writer.add_figure(data_loader_name + os.path.join(name[y_index], str(i), str(series_index), str(y_series.shape[0])), fig, global_step=epoch)
                    plt.close()
                    fig.clf()


        if plt_visualize and pre_x.shape[1]>=3:
            plt.figure(figsize=(10, 6))

            for y_index in range(len(config.Target_Col)):
                for series_index in range(min(config.batch_size, 3)):
                    plt.subplot('33'+str(series_index+y_index*3+1))
                    plt.plot(y_series[:, series_index, y_index])
                    plt.plot(y_est_series[:, series_index, y_index])
                    plt.legend(['Real', 'forecast'])
                    plt.title('{}-{}-{}'.format(name[y_index],i,series_index))

            plt.show()

    total_test_loss /= total_test_items

    used_time['evaluation and visualization'] += time.time() - time_point
    return total_test_loss, total_test_items, used_time

def train_net(net, train_loader, val_loader, test_loader, logging, config):

    train_loss_list = []
    val_RRSE_list = []

    use_cuda = config.use_cuda if torch.cuda.is_available() else False
    if use_cuda:
        net = net.cuda()


    if config.loss_func == 'L2':
        critic = torch.nn.MSELoss()
    elif config.loss_func == 'gauss':
        from common import GaussLoss
        critic = GaussLoss(len(config.Target_Col), config)
        sigma_optim = torch.optim.Adam([critic.sigma])


    optim = torch.optim.Adam(net.parameters(), lr=config.lr)
    schedualer = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.95, last_epoch=-1)

    from common import MyWriter

    writer = MyWriter(
        save_path=config.tb_path,
        is_write=config.tb

    )
    from collections import defaultdict
    best_RRSE = defaultdict(lambda : 1e8)
    best_info = (-1,1e8, 1e8, 1e8)

    best_ckpt = {
        'optim': optim.state_dict(),
        'scaler_mean': config.scaler.mean_,
        'scaler_var': config.scaler.var_,
        'config': copy.deepcopy(config)
    }

    epoch_stable = 0

    for epoch in range(0, config.epochs):

        net.train()

        start_time = time.time()
        total_train_loss = 0
        total_train_items = 0
        for i, data in enumerate(train_loader):
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
            start_time = time.time()
            y_estimate_all = net((pre_x, pre_y, forward_x))
            loss = critic(y_estimate_all, forward_y)
            total_train_loss += float(loss)*pre_x.shape[1]
            total_train_items += pre_x.shape[1]

            optim.zero_grad()

            if config.loss_func == 'gauss':
                sigma_optim.zero_grad()


            loss.backward()

            optim.step()
            if config.loss_func == 'gauss':
                sigma_optim.step()

            # write to tensorboard
            process_time = time.time() - start_time

            # bar.set_description("compute efficiency: {:.2f}%, process time: {:.2f}s, epoch: {}/{}".format(
            #     100*process_time/(process_time+prepare_time),process_time ,epoch, config.epochs)
            # )
            logging("epoch: {}/{}, batch: {}/{}, train loss: {}, cal times: {} compute efficiency: {:.2f}%, process time: {:.2f}s, ".format(
                epoch, config.epochs, i, len(train_loader),float(loss), net.ode_net.cell.call_times,
                100*process_time/(process_time+prepare_time),process_time )
            )
            #print("compute efficiency: {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time), epoch, config.epochs))
            start_time = time.time()




        writer.add_scalar('train_loss', total_train_loss/total_train_items, epoch)
        if config.loss_func =='gauss':
            for i in range(critic.sigma.shape[0]):
                writer.add_scalar('sigma'+str(i), float(critic.get_cov()[i,i].cpu()), epoch)
        train_loss_list.append(total_train_loss/total_train_items)
        #print('train loss after {} epoch is {:.2f}'.format(epoch+1, total_train_loss/total_train_items))
        schedualer.step()

        if epoch % config.test_period == config.test_period - 1 or epoch == 0:

            val_rrse, test_rrse = {}, {}
            val_mse, test_mse = {}, {}
            with torch.no_grad():
                for i, forward_length in enumerate(config.test_look_forward):
                    # val_error,_, used_time = test_net(net, val_loader[i], config, writer, {'RRSE': RRSE, 'MSE': torch.nn.MSELoss()}, epoch=epoch,
                    #                           plt_visualize=config.plt_visualize, tb_visualize=config.tb)

                    val_error,_, used_time = test_net(net, val_loader[i], config, writer, {'RRSE': RRSE}, epoch=epoch,
                                                     plt_visualize=config.plt_visualize, tb_visualize=config.tb)
                    val_rrse[str(forward_length)] = val_error['RRSE']
                    #val_mse[str(forward_length)] = val_error['MSE']

                    # test_error, _, _ = test_net(net, test_loader[i], config, writer, {'RRSE': RRSE, 'MSE': torch.nn.MSELoss()}, epoch=epoch,
                    #                                      plt_visualize=config.plt_visualize, tb_visualize=config.tb)

                    test_error, _, _ = test_net(net, test_loader[i], config, writer, {'RRSE': RRSE}, epoch=epoch,
                                                plt_visualize=config.plt_visualize, tb_visualize=config.tb)
                    test_rrse[str(forward_length)] = test_error['RRSE']
                    #test_mse[str(forward_length)] = test_error['MSE']

                    # 'forward': 0.0,
                    # 'prepare data': 0.0,
                    # 'evaluation and visualization': 0.0
                    sum_time = sum(used_time.values())
                    logging('Test in Epoch {:04d} | forward len {:04d} | call times {} | Time {:.2f} \
                            | pre {:.2f}-{:.1f}% | forward {:.2f}-{:.1f}% | eval {:.2f}-{:.1f}% \
                            | val error{:.2f} | test error {:.2f}'.format(
                        epoch, forward_length, net.ode_net.cell.call_times,sum_time, used_time['prepare data'], 100*used_time['prepare data']/sum_time,
                        used_time['forward'], 100*used_time['forward']/sum_time,
                        used_time['evaluation and visualization'], 100*used_time['evaluation and visualization']/sum_time,
                        val_error['RRSE'], test_error['RRSE']
                    ))


                # total_val_metrics, _, acc_time = test_net(net, val_loader, config, writer, {'RRSE': RRSE, 'MSE': torch.nn.MSELoss()}, epoch=epoch,
                #                               plt_visualize=config.plt_visualize, tb_visualize=config.tb)

            #print('val loss after {} epoch is {:.2f}'.format(epoch+1, total_val_loss))

            writer.add_scalars('val_RRSE', val_rrse, epoch)
            writer.add_scalars('test_RRSE', test_rrse, epoch)

            # writer.add_scalars('val_MSE', val_mse, epoch)
            # writer.add_scalars('test_MSE', test_mse, epoch)
            #writer.add_scalar('val_MSE', total_val_MSE, epoch)
            writer.add_scalar('val_time', sum_time, epoch)
            val_RRSE_list.append(val_rrse)

            # logging('\nepoch %04d | loss %.3f | val rrse %.3f | test rrse %.3f '%
            #         (epoch, total_train_loss/total_train_items, val_rrse[config.cmp_length], test_rrse[config.cmp_length]))

            update_ckpt = False
            for i, cmp_length in enumerate(config.test_look_forward):
                cmp_length = str(cmp_length)
                if val_rrse[cmp_length] < best_RRSE[cmp_length]:
                    best_RRSE[cmp_length] = val_rrse[cmp_length]
                    best_info = (epoch, total_train_loss/total_train_items, val_rrse[str(config.cmp_length)], test_rrse[str(config.cmp_length)])
                    best_ckpt['net'+str(cmp_length)] = copy.deepcopy(net.state_dict())
                    best_ckpt['epoch'+str(cmp_length)] = epoch
                    update_ckpt = True
            if update_ckpt:
                epoch_stable = 0
            else:
                epoch_stable += 1
            if epoch_stable > 10:
                logging('Early stop in epoch %04d.' % epoch)
                break

    if not os.path.exists('./ckpt/'+config.save_dir):
        os.makedirs('./ckpt/'+config.save_dir)
    torch.save(best_ckpt, os.path.join('ckpt', config.save_dir, 'best')+'.pth')
    logging('best loss = {:.4f} in epoch = {} with train_loss = {:.4f} with  val rrse = {:.4f} with test rrse = {:.4f}'.format(
        best_RRSE[str(config.cmp_length)], best_info[0], best_info[1], best_info[2], best_info[3]
    ))
    return net, train_loss_list, val_RRSE_list

