import torch
from tqdm import tqdm
from utils.net_frame import try_gpu
from torch import nn
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision

# 定义训练函数
def train(net,trainer,train_iter,scheduler,loss_fn,lr,num_epochs,devices_idx = None):
    """支持多卡的训练"""
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

def show_loss(data_path,save_path):
    plt_loss = joblib.load(data_path)
    plt.plot(np.arange(len(plt_loss)),plt_loss,label = "Train_Loss",marker = 'o')
    plt.legend()
    plt.savefig(save_path)

def preprocess(img_path,transforms):
    """图像预处理"""
    mode = torchvision.io.image.ImageReadMode.RGB
    img = torchvision.io.read_image(img_path,mode)
    img = img.float() / 255 # 浮点化、归一化
    input_tensor = transforms(img)
    input_tensor = input_tensor.unsqueeze(0) # 添加批量维度
    return input_tensor

def model_infer(net,input_tensor,device_idx = None):
    """模型推理"""
    if device_idx == None:
        device = try_gpu(i = 0)
    else:
        device = try_gpu(device_idx)
    net.eval()
    net.to(device)
    input_tensor = input_tensor.to(device)
    out_tensor = net(input_tensor)
    return out_tensor

def nms(boxes, scores,threshold):
    """非极大值抑制"""
    x1 = boxes[:, 0]  # [n,]
    y1 = boxes[:, 1]  # [n,]
    x2 = boxes[:, 2]  # [n,]
    y2 = boxes[:, 3]  # [n,]
    areas = (x2 - x1) * (y2 - y1)  # [n,]

    _, ids_sorted = scores.sort(0, descending=True)  # [n,]
    ids = []
    while ids_sorted.numel() > 0:
        # Assume `ids_sorted` size is [m,] in the beginning of this iter.

        i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
        ids.append(i)

        if ids_sorted.numel() == 1:
            break  # If only one box is left (i.e., no box to supress), break.

        inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i])  # [m-1, ]
        inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i])  # [m-1, ]
        inter_x2 = x2[ids_sorted[1:]].clamp(max=x2[i])  # [m-1, ]
        inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i])  # [m-1, ]
        inter_w = (inter_x2 - inter_x1).clamp(min=0)  # [m-1, ]
        inter_h = (inter_y2 - inter_y1).clamp(min=0)  # [m-1, ]

        inters = inter_w * inter_h  # intersections b/w/ box `i` and other boxes, sized [m-1, ].
        unions = areas[i] + areas[ids_sorted[1:]] - inters  # unions b/w/ box `i` and other boxes, sized [m-1, ].
        ious = inters / unions  # [m-1, ]

        # Remove boxes whose IoU is higher than the threshold.
        ids_keep = (ious <= threshold).nonzero().squeeze()  # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
        if ids_keep.numel() == 0:
            break  # If no box left, break.
        ids_sorted = ids_sorted[ids_keep + 1]  # `+1` is needed because `ids_sorted[0] = i`.

    return torch.LongTensor(ids)


def decode(pred_tensor,grid_size,num_bboxes,conf_thresh,prob_thresh,nb_classes):
    """与数据dataset的encode对应的decode"""
    S, B, C = grid_size,num_bboxes,nb_classes
    boxes, labels, confidences, class_scores = [], [], [], []

    cell_size = 1.0 / float(S)

    pred_tensor = pred_tensor.cpu().data.squeeze(0)

    pred_tensor_conf_list = []
    for b in range(B):
        pred_tensor_conf_list.append(pred_tensor[:, :, 5 * b + 4].unsqueeze(2))
    grid_ceil_conf = torch.cat(pred_tensor_conf_list, 2)

    # 每个grid的候选框确定保留的是哪个(关键)
    grid_ceil_conf, grid_ceil_index = grid_ceil_conf.max(2)
    class_conf, class_index = pred_tensor[:, :, 5 * B:].max(2)
    class_conf[class_conf <= conf_thresh] = 0
    class_prob = class_conf * grid_ceil_conf

    for i in range(S):
        for j in range(S):
            if float(class_prob[j, i]) < prob_thresh:
                continue
            box = pred_tensor[j, i, 5 * grid_ceil_index[j, i]: 5 * grid_ceil_index[j, i] + 4]
            xy_start_pos = torch.FloatTensor([i, j]) * cell_size
            xy_normalized = box[:2] * cell_size + xy_start_pos
            wh_normalized = box[2:]
            box_xyxy = torch.FloatTensor(4)
            box_xyxy[:2] = xy_normalized - 0.5 * wh_normalized
            box_xyxy[2:] = xy_normalized + 0.5 * wh_normalized

            boxes.append(box_xyxy)
            labels.append(class_index[j, i])
            confidences.append(grid_ceil_conf[j, i])
            class_scores.append(class_conf[j, i])

    if len(boxes) > 0:
        boxes = torch.stack(boxes, 0)
        labels = torch.stack(labels, 0)
        confidences = torch.stack(confidences, 0)
        class_scores = torch.stack(class_scores, 0)
    else:
        boxes = torch.FloatTensor(0, 4)
        labels = torch.LongTensor(0)
        confidences = torch.FloatTensor(0)
        class_scores = torch.FloatTensor(0)

    return boxes, labels, confidences, class_scores

def postprocess(output,width, height,VOC_CLASSES,grid_size,num_bboxes,conf_thresh,prob_thresh,nms_thresh,nb_classes):

    output = output.permute(0,2,3,1) # (N,C,H,W) -> (N,H,W,C)

    boxes,labels,probs = [],[],[]

    boxes_list, labels_list, confidences_list, class_scores_list = decode(output, grid_size, num_bboxes,
                                                                          conf_thresh, prob_thresh,
                                                                          nb_classes)
    if boxes_list.shape[0] != 0:
        boxes_nms, labels_nms, probs_nms = [], [], []
        for class_label in range(len(VOC_CLASSES)):
            ids = (labels_list == class_label)
            if torch.sum(ids) == 0:
                continue

            boxes_list_current_cls = boxes_list[ids]
            labels_list_current_cls = labels_list[ids]
            confidences_list_current_cls = confidences_list[ids]
            class_scores_list_current_cls = class_scores_list[ids]

            ids_postprocess = nms(boxes_list_current_cls, confidences_list_current_cls, nms_thresh)

            boxes_nms.append(boxes_list_current_cls[ids_postprocess])
            labels_nms.append(labels_list_current_cls[ids_postprocess])
            probs_nms.append(
                confidences_list_current_cls[ids_postprocess] * class_scores_list_current_cls[ids_postprocess])

        boxes_nms = torch.cat(boxes_nms, 0)
        labels_nms = torch.cat(labels_nms, 0)
        probs_nms = torch.cat(probs_nms, 0)

        for box, label, prob in zip(boxes_nms, labels_nms, probs_nms):
            x1, x2 = width * box[0], width * box[2]  # unnormalize x with image width.
            y1, y2 = height * box[1], height * box[3]  # unnormalize y with image height.
            boxes.append(((x1, y1), (x2, y2)))

            label_idx = int(label)  # convert from LongTensor to int.
            class_name = VOC_CLASSES[label_idx]
            labels.append(class_name)

            prob = float(prob)
            probs.append(prob)

    return boxes,labels,probs