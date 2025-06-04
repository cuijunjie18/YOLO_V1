import torch
import torchvision
import argparse
from torch import nn
from tqdm import tqdm
from utils.net_frame import *
from utils.loss import YoloLoss
from utils.datasets import YoloData
from modules.yolov1 import Yolov1

def get_args_parser():
    parser = argparse.ArgumentParser('YOLOV1 for train', add_help = False)
    parser.add_argument('--',default = "VOC2012/Annotations",
                        type = str,help = "Path to raw xml-files.")
    parser.add_argument('--save_folder',default = "datasets",
                        type = str,help = "Path to save file converted.")
    return parser

# 定义训练函数
def train(net,trainer,train_iter,scheduler,loss_fn,lr,num_epochs,devices_idx = None):
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
            loop.set_postfix({"LOSS" : loss_temp / total_nums,"lr" : "{:e}".format(scheduler.get_last_lr()[0])})
        scheduler.step()
        loss_plt.append(loss_temp / total_nums)
    return loss_plt

if __name__ == "__main__":
    # parser = get_args_parser()
    # args = parser.parse_args()
    # 定义图像transforms
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225]
        ),torchvision.transforms.Resize((448,448))]
    )

    # 加载数据集
    yolodata = YoloData("datasets/JPEGImages","datasets/train.txt",transforms = transforms)

    batch_size = 32
    train_iter = data.DataLoader(yolodata,batch_size,shuffle = True)

    net = Yolov1()

    num_epochs = 150
    lr = 5e-4
    loss = YoloLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer,step_size = num_epochs * 0.4,gamma = 0.1)

    plt_loss = train(net,trainer,train_iter,scheduler,loss_fn = loss,
                 num_epochs = num_epochs,lr = lr,devices_idx = [7])

    import joblib
    import os
    save_prefix = "results/exp4"
    os.makedirs(save_prefix,exist_ok = True)
    joblib.dump(plt_loss,os.path.join(save_prefix,"plt_loss.joblib"))
    torch.save(net,os.path.join(save_prefix,"best.pt"))