import torch
from torch import nn

def get_yolov1(num_classes = 20):
    return nn.Sequential(
        nn.Conv2d(3,64,kernel_size = 7,stride = 2),nn.LeakyReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(64,192,kernel_size = 3),nn.LeakyReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(192,128,1),nn.LeakyReLU(),
    )


# class yolov1(nn.modules):
#     """yolov1网络实现"""

#     def __init__(self):
#         self.