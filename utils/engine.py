import torch
from tqdm import tqdm
from utils.net_frame import try_gpu
from torch import nn
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

# 定义训练函数
def train(net,trainer,train_iter,test_iter,loss_fn,lr,num_epochs,devices_idx = None):
    """训练情感分析模型"""
    # 设置设备
    if devices_idx == None:
        device = try_gpu(i = 0)
    else:
        assert (type(devices_idx == list) and 
                type(devices_idx[0]) == int),"devices_idx must be list of int"
        devices = [torch.device(f"cuda:{i}")
                   for i in devices_idx]
    print(f"Training on{devices}")
    
    # 多GPU加载网络(当len(devices) == 1时即单卡训练)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    # 开始训练
    loss_plt = []
    train_accs = []
    test_accs = []
    for epoch in range(num_epochs):
        net.train() # 循环涉及评估，则每次循环前要net.train()
        loop = tqdm(train_iter,desc = f"Epoch:[{epoch + 1}/{num_epochs}]",
                    total = len(train_iter))
        loss_temp = 0
        total_nums = 0
        for batch in loop:
            # 清空梯度
            trainer.zero_grad()

            # forward
            X,Y = batch
            if type(X) == list:
                X = [x.to(devices[0]) for x in X]
                total_nums += X[0].shape[0]
            else:
                X = X.to(devices[0]) # 放置在devices[0]即可
                total_nums += X.shape[0]
            Y = Y.to(devices[0])
            # print(X.shape,Y.shape)
            y_pred = net(X)

            # count loss and backwar
            loss = loss_fn(y_pred,Y)
            loss.sum().backward()
            trainer.step()

            # 先step后再调用item()，否则切断计算图
            loss_temp += loss.sum().item()
            
            # # update parameters
            # trainer.step()
            loop.set_postfix({"LOSS" : loss_temp / total_nums,"lr" : "{:e}".format(trainer.param_groups[0]['lr'])})
        loss_plt.append(loss_temp / total_nums)
    return loss_plt

def show_loss(data_path,save_path):
    plt_loss = joblib.load(data_path)
    plt.plot(np.arange(len(plt_loss)),plt_loss,label = "Train_Loss",marker = 'o')
    plt.legend()
    plt.savefig(save_path)