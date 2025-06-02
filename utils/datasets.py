import torch
import torchvision
from torch.utils import data
from tqdm import tqdm
import os

class YoloData(data.Dataset):
    """加载Yolo目标检测数据集"""
    def __init__(self,img_dir,label_file,num_classes = 20,
                 grid_size = 7,num_bboxes = 2,transforms = None):
        self.img_dir = img_dir
        self.label_file = label_file
        self.transforms = transforms # 图像变化

        self.normalize_transform = torchvision.transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225]
        ) # RGB格式的通道归一化参数

        self.images = []
        self.boxes = []
        self.labels = []

        # 对应论文里的S、B、C
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

        mode = torchvision.io.image.ImageReadMode.RGB # RGB格式

        # 解析数据集
        file = open(self.label_file,"r",encoding = 'utf-8')
        lines = file.readlines()
        loop = tqdm(lines,desc = "Extract dataset",total = len(lines))
        for line in loop:
            infomation = line.strip().split()
            self.images.append(torchvision.io.read_image(
                os.path.join(img_dir,infomation[0]),mode
            ))
            num_boxes = (len(infomation) - 1) // 5
            boxes = []
            labels = []
            for i in range(num_boxes):
                xmin = float(infomation[1 + 5 * i])
                ymin = float(infomation[2 + 5 * i])
                xmax = float(infomation[3 + 5 * i])
                ymax = float(infomation[4 + 5 * i])
                cls = float(infomation[5 + 5 * i])
                boxes.append([xmin,ymin,xmax,ymax])
                labels.append(cls)
            self.boxes.append(torch.tensor(boxes))
            self.labels.append(torch.LongTensor(labels)) # label为long
        file.close()

        # 原始数据集编码为可计算的形式
        self.raw_size = [(img.shape[1],img.shape[2]) for img in self.images]
        self.features = [self.img_transform(img) for img in 
                         tqdm(self.images,desc = 'Normalize img',total = len(self.images))]
        for i in tqdm(range(len(self.boxes)),desc = "Normalize bboxes",
                      total = len(self.boxes)):
            H,W = self.raw_size[i] # 注意这里用的是原来的比例(resize前的)
            self.boxes[i] /= torch.Tensor([W,H,W,H]).expand_as(self.boxes[i])
        self.targets = []
        for i in tqdm(range(len(self.boxes)),desc = "Generating targets",
                      total = len(self.boxes)):
            self.targets.append(self.encode(self.boxes[i],self.labels[i]))

    def encode(self,boxes,labels):
        """生成模型可计算的监督数据"""
        cell_size = 1.0 / self.S
        target = torch.zeros((self.S,self.S,self.B * 5 + self.C)) #(H,W,C)
        boxes_wh = boxes[:,2:4] - boxes[:,:2]
        boxes_xy = (boxes[:,2:4] + boxes[:,:2]) / 2.0
        for xy,wh,label in zip(boxes_xy,boxes_wh,labels):
            ij = (xy / cell_size).ceil() - 1.0
            i, j = int(ij[0]),int(ij[1])
            x0y0 = ij * cell_size
            xy_offset = (xy - x0y0) / cell_size # 计算相对格子的偏移量
            for b in range(self.B):
                s = b * 5
                target[j,i,s:s+2] = xy_offset
                target[j,i,s+2:s+4] = wh
                target[j,i,s + 4] = 1.0 # 置信度为1
            target[j,i,self.B * 5 + label] = 1.0 # 对应类别的概率为1
        return target.permute(2,0,1) #(C,H,W)

    def img_transform(self,img):
        """图像transform,可能包含各种增广及变化操作"""
        img = img.float() / 255
        if isinstance(self.transforms,torchvision.transforms.Compose):
            return self.transforms(img)
        elif isinstance(self.transforms,list):
            for transform in self.transforms:
                img = transform(img)
            return img

    def normalize_img(self,img):
        """图像归一化"""
        img = img.float()
        return self.transform(img.float() / 255)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.features[index],self.targets[index]
    
if __name__ == "__main__":
    """调试模式"""

    # 定义图像transforms
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225]
        ),torchvision.transforms.Resize((448,448))]
    )

    # transforms = [
    #     torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]),
    #     torchvision.transforms.Resize(448)
    # ]

    # 加载数据集
    yolodata = YoloData("datasets/JPEGImages","datasets/train.txt",transforms = transforms)

    # 查看关键属性的信息
    print(len(yolodata.labels),yolodata.labels[0].shape,type(yolodata.labels[0]))
    print(len(yolodata.features),yolodata.features[0].shape,type(yolodata.features[0]))
    print(len(yolodata.boxes),yolodata.boxes[0].shape,type(yolodata.boxes[0]))
    print(len(yolodata.targets),yolodata.targets[0].shape,type(yolodata.targets[0]))

    # 测试迭代器
    train_iter = data.DataLoader(yolodata,batch_size = 64)
    for X,Y in train_iter:
        print(X.shape)
        print(Y.shape)
        break