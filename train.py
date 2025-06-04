import torch
import torchvision
import argparse
from torch import nn
from tqdm import tqdm
from utils.net_frame import *
from utils.loss import YoloLoss
from utils.datasets import YoloData
from modules.yolov1 import Yolov1
from utils.engine import train
import os

def get_args_parser():
    parser = argparse.ArgumentParser('YOLOV1 for train', add_help = False)
    parser.add_argument('--data_dir',default = "./datasets",type = str,
                        help = "Path to dataset.")
    parser.add_argument('--batch_size',default = 32,type = int,
                        help = "Batch size for one train iteration.")
    parser.add_argument('--num_epochs',default = 10,type = int,
                        help = "Epochs for training.")
    parser.add_argument('--lr',default = 5e-4,type = float,
                        help = "Learning rate for training.")
    parser.add_argument('--devices_idx',default = [0],nargs = '+',type = int,
                        help = "Training cuda devices index list.")
    parser.add_argument('--save_dir',default = 'results/exp',type = str,
                        help = "save folder for training.")
    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # 获取设备列表
    devices_idx = args.devices_idx
    print(type(devices_idx),devices_idx)

    # 定义图像transforms
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225]
        ),torchvision.transforms.Resize((448,448))]
    )

    # 加载数据集
    img_dir = os.path.join(args.data_dir,"JPEGImages")
    annotations = os.path.join(args.data_dir,"train.txt")
    yolodata = YoloData(img_dir,annotations,transforms = transforms)

    batch_size = args.batch_size
    train_iter = data.DataLoader(yolodata,batch_size,shuffle = True)

    net = Yolov1()

    num_epochs = 150
    lr = 5e-4
    loss = YoloLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer,step_size = num_epochs * 0.4,gamma = 0.1)

    plt_loss = train(net,trainer,train_iter,scheduler,loss_fn = loss,
                 num_epochs = num_epochs,lr = lr,devices_idx = devices_idx)

    import joblib
    import os
    save_prefix = args.save_dir
    os.makedirs(save_prefix,exist_ok = True)
    joblib.dump(plt_loss,os.path.join(save_prefix,"plt_loss.joblib"))
    torch.save(net,os.path.join(save_prefix,"best.pt"))