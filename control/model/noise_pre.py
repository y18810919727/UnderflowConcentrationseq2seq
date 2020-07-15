from config import args as config
import pandas
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime  # 用于计算时间

from control.scaler import MyScaler

col = [1, 6]
# 定义常量
INPUT_SIZE = 2  # 定义输入的特征数
HIDDEN_SIZE = 8    # 定义一个LSTM单元有多少个神经元
BATCH_SIZE = 16   # batch
EPOCH = 1000    # 学习次数
LR = 0.001   # 学习率
DROP_RATE = 0.2    #  drop out概率
LAYERS = 1         # LSTM层数
MODEL = 'LSTM'     # 模型名字


#将数据存储为两个矩阵，一个矩阵的ind位置存储t时刻的值，另一个矩阵存储t+1时刻的值
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    i = 0
    while(i < len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back: i + look_back + 2, :])
        i = i + 2
    return np.array(dataX), np.array(dataY)

def load_data():
    # load data
    state_dic = torch.load('./ckpt/rnn_ode_affine_3_4_cubic_full/best.pth')
    model_config = state_dic['config']
    _mean, _var = state_dic['scaler_mean'], state_dic['scaler_var']
    my_scaler = MyScaler(_mean, _var, model_config.all_col, model_config.Target_Col, model_config.Control_Col,
                         config.controllable,
                         config.uncontrollable)

    ori_data = np.array(pandas.read_csv(model_config.DATA_PATH))
    scaled_data = my_scaler.scale_all(ori_data)[:, col]
    return scaled_data


class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            dropout=DROP_RATE,
            batch_first=True  # True-(batch, seq_len, feature)
        )
        self.hidden_out = nn.Linear(HIDDEN_SIZE, INPUT_SIZE * 2)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        r_out, (h_s, h_c) = self.rnn(x)  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output


if __name__ == '__main__':

    scaled_data = load_data()
    # 设置GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 设置随机种子
    torch.manual_seed(0)

    dataX, dataY = create_dataset(scaled_data, look_back=15)

    X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.3, random_state=42)
    # X_valid = X_test[:int(len(X_test)*0.5), :]
    # Y_valid = Y_test[:int(len(X_test) * 0.5), :]
    X_valid = X_test
    Y_valid = Y_test
    X_test = X_test[int(len(X_test)*0.5):, :]
    Y_test = Y_test[int(len(X_test) * 0.5):, :]
    # 使用list只会把最外层变为list，内层还是ndarray，和.tolist()方法不同
    data_train = list(X_train)
    data_valid = list(X_valid)
    data_test = list(X_test)

    data_train = list(zip(data_train, list(Y_train)))  # 最外层是list，次外层是tuple，内层都是ndarray
    data_valid = list(zip(data_valid, list(Y_valid)))
    data_test = list(zip(data_test, list(Y_test)))

    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=False)

    lstm = lstm().to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)  # optimize all cnn parameters
    # loss_func = nn.CrossEntropyLoss()
    loss_func = torch.nn.MSELoss(reduce=True, size_average=True)
    # 定义学习率衰减点，训练到50%和75%时学习率缩小为原来的1/10
    mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCH // 2, EPOCH // 4 * 3], gamma=0.1)

    train_loss = []
    valid_loss = []
    min_valid_loss = np.inf
    for i in range(EPOCH):
        total_train_loss = []
        lstm.train()  # 进入训练模式
        for step,(b_x, b_y) in enumerate(train_loader):
            # lr = set_lr(optimizer, i, EPOCH, LR)
            b_x = b_x.type(torch.FloatTensor).to(device)
            b_y = b_y.type(torch.FloatTensor).to(device).view(b_y.shape[0], INPUT_SIZE*2)
            prediction = lstm(b_x)
            # x = prediction[:, -1, :]
            # y = b_y.view(b_y.size()[0])
            loss = loss_func(prediction[:, -1, :], b_y)
            # print('step_'+str(step)+':'+str(loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss))
        # print('step_'+str(i)+'_train_loss:'+str(np.mean(total_train_loss)))

        total_valid_loss = []
        lstm.eval()
        for step, (b_x, b_y) in enumerate(valid_loader):
            b_x = b_x.type(torch.FloatTensor).to(device)
            b_y = b_y.type(torch.FloatTensor).to(device).view(b_y.shape[0], INPUT_SIZE*2)
            with torch.no_grad():
                prediction = lstm(b_x)  # rnn output
            loss = loss_func(prediction[:, -1, :], b_y)  # calculate loss
            total_valid_loss.append(loss.item())
        valid_loss.append(np.mean(total_valid_loss))
        # print('step_' + str(i) + '_valid_loss:' + str(np.mean(total_valid_loss)))

        if (np.mean(total_valid_loss) < min_valid_loss):
            torch.save(lstm.state_dict(), './ckpt/noise_pre/best.pth')  # 保存字典对象，里面'model'的value是模型
            min_valid_loss = np.mean(total_valid_loss)

        if i % 50 == 0 and i != 0:
            print('train_loss:'+str(train_loss))
            print('valid_loss:'+str(valid_loss))
            plt.plot(train_loss, label='train_loss')
            plt.plot(valid_loss, label='valid_loss')
            plt.xlabel('step')
            plt.ylabel('MSE_loss')
            plt.legend()
            plt.show()

        # 编写日志
        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                      'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), EPOCH,
                                                                      train_loss[-1],
                                                                      valid_loss[-1],
                                                                      min_valid_loss,
                                                                      optimizer.param_groups[0]['lr'])
        mult_step_scheduler.step()  # 学习率更新
        # 服务器一般用的世界时，需要加8个小时，可以视情况把加8小时去掉
        # print(str(datetime.datetime.now() + datetime.timedelta(hours=8)) + ': ')
        print(log_string)  # 打印日志


