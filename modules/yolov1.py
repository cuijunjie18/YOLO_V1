import torch
from torch import nn
import math
from torchinfo import summary

def get_yolov1(num_classes = 20,num_bboxes = 2):
    """获取yolov1模型"""
    return nn.Sequential(
        nn.Conv2d(3,64,kernel_size = 7,stride = 2,padding = 3),nn.LeakyReLU(),
        nn.MaxPool2d(2,2),                    # k = 2,s = 2的MaxPool2d层使图像分辨率减半
        nn.Conv2d(64,192,kernel_size = 3,padding = 1),nn.LeakyReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(192,128,1),nn.LeakyReLU(),
        nn.Conv2d(128,256,3,padding = 1),nn.LeakyReLU(),
        nn.Conv2d(256,256,1),nn.LeakyReLU(),
        nn.Conv2d(256,512,3,padding = 1),nn.LeakyReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(512,256,1),nn.LeakyReLU(),
        nn.Conv2d(256,512,3,padding = 1),nn.LeakyReLU(),
        nn.Conv2d(512,256,1),nn.LeakyReLU(),
        nn.Conv2d(256,512,3,padding = 1),nn.LeakyReLU(),
        nn.Conv2d(512,256,1),nn.LeakyReLU(),
        nn.Conv2d(256,512,3,padding = 1),nn.LeakyReLU(),
        nn.Conv2d(512,256,1),nn.LeakyReLU(),
        nn.Conv2d(256,512,3,padding = 1),nn.LeakyReLU(),
        nn.Conv2d(512,512,1),nn.LeakyReLU(),
        nn.Conv2d(512,1024,3,padding = 1),nn.LeakyReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(1024,512,1),nn.LeakyReLU(),
        nn.Conv2d(512,1024,3,padding = 1),nn.LeakyReLU(),
        nn.Conv2d(1024,512,1),nn.LeakyReLU(),
        nn.Conv2d(512,1024,3,padding = 1),nn.LeakyReLU(),
        nn.Conv2d(1024,1024,3,padding = 1),nn.LeakyReLU(),
        nn.Conv2d(1024,1024,3,stride = 2,padding = 1),nn.LeakyReLU(),
        nn.Conv2d(1024,1024,3,padding = 1),nn.LeakyReLU(),
        nn.Conv2d(1024,1024,3,padding = 1),nn.LeakyReLU(),
        nn.Flatten(),nn.Linear(7 * 7 * 1024,4096),
        nn.Linear(4096,7 * 7 * (num_bboxes * 5 + num_classes))
    )

class Yolov1(nn.Module):
    def __init__(self,num_classes = 20,num_bboxes = 2):
        super().__init__()
        self.B = num_bboxes
        self.C = num_classes
        self.layer = get_yolov1(self.C,self.B)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        
    def forward(self,X):
        X = self.layer(X)
        X = X.reshape(X.shape[0],self.B * 5 + 
                      self.C,7,7)
        return X
    
if __name__ == "__main__":
    """调试模式"""
    model = yolov1()
    conv_nums = 0
    for m in model.modules():
        if (type(m) == nn.Conv2d):
            conv_nums += 1
    print(f"Conv Layers:{conv_nums}")

    net = yolov1()
    input = torch.rand((1,3,448,448))
    print(summary(net,input_data = input))
