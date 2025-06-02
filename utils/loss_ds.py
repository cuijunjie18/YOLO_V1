import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Detect_Loss(nn.Module):
    def __init__(self, feature_size=7, num_bboxes=2, num_classes=20, lambda_coord=5.0, lambda_noobj=0.5):
        """
        初始化 YOLOv1 损失函数
        
        参数:
        - feature_size: 特征图尺寸 (默认7x7)
        - num_bboxes: 每个网格预测的边界框数量 (默认2)
        - num_classes: 类别数量 (默认20)
        - lambda_coord: 坐标损失的权重系数 (默认5.0)
        - lambda_noobj: 无目标置信度损失的权重系数 (默认0.5)
        """
        super(Detect_Loss, self).__init__()
        
        self.S = feature_size       # 特征图大小 (SxS 网格)
        self.B = num_bboxes         # 每个网格预测的边界框数量
        self.C = num_classes        # 类别数量
        self.lambda_coord = lambda_coord  # 坐标损失权重
        self.lambda_noobj = lambda_noobj  # 无目标置信度损失权重

    def compute_iou(self, bbox1, bbox2):
        """
        计算两组边界框之间的交并比(IoU)
        
        参数:
        - bbox1: 形状为 [N, 4] 的边界框 (xmin, ymin, xmax, ymax)
        - bbox2: 形状为 [M, 4] 的边界框 (xmin, ymin, xmax, ymax)
        
        返回:
        - iou: 形状为 [N, M] 的 IoU 矩阵
        """
        # 获取边界框数量
        N = bbox1.size(0)
        M = bbox2.size(0)
        
        # 计算交集的左上角坐标 (left-top)
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)   # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        
        # 计算交集的右下角坐标 (right-bottom)
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)    # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        
        # 计算交集的宽高
        wh = rb - lt
        wh[wh < 0] = 0  # 处理无重叠的情况
        inter = wh[:, :, 0] * wh[:, :, 1]  # 交集面积 [N, M]
        
        # 计算两个边界框各自的面积
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])  # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])  # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M, ] -> [1, M] -> [N, M]
        
        # 计算并集面积
        union = area1 + area2 - inter  # [N, M]
        
        # 计算 IoU
        iou = inter / union  # [N, M]
        
        return iou

    def forward(self, pred_tensor, target_tensor):
        """
        计算 YOLOv1 损失
        
        参数:
        - pred_tensor: 模型预测的输出张量，形状为 [batch_size, S, S, B*5 + C]
        - target_tensor: 目标标签张量，形状与 pred_tensor 相同
        
        返回:
        - loss: 计算得到的损失值
        """
        # 获取参数
        S, B, C = self.S, self.B, self.C
        N = 5 * B + C  # 每个网格的预测值总数
        
        batch_size = pred_tensor.size(0)
        
        # 创建目标存在和不存在的位置掩码
        # target_tensor[:, :, :, 4] 是第一个边界框的置信度
        coord_mask = target_tensor[:, :, :, 4] > 0   # 有目标的网格位置
        noobj_mask = target_tensor[:, :, :, 4] == 0  # 无目标的网格位置
        
        # 扩展掩码维度以匹配目标张量
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)
        
        # 提取有目标位置的预测值和目标值
        coord_pred = pred_tensor[coord_mask].view(-1, N)
        coord_target = target_tensor[coord_mask].view(-1, N)
        
        # 分割边界框预测和类别预测
        bbox_pred = coord_pred[:, :5 * B].contiguous().view(-1, 5)
        class_pred = coord_pred[:, 5 * B:]
        
        # 分割边界框目标和类别目标
        bbox_target = coord_target[:, :5 * B].contiguous().view(-1, 5)
        class_target = coord_target[:, 5 * B:]
        
        # 提取无目标位置的预测值和目标值
        noobj_pred = pred_tensor[noobj_mask].view(-1, N)
        noobj_target = target_tensor[noobj_mask].view(-1, N)
        
        # 处理无目标位置的置信度损失
        # 创建掩码以选择所有边界框的置信度 (位置 4, 9, ...)
        noobj_conf_mask = torch.cuda.BoolTensor(noobj_pred.size()).fill_(0)
        for b in range(B):
            noobj_conf_mask[:, 4 + b * 5] = 1  # 设置每个边界框的置信度位置
            
        # 提取无目标位置的置信度预测值和目标值
        noobj_pred_conf = noobj_pred[noobj_conf_mask]
        noobj_target_conf = noobj_target[noobj_conf_mask]
        
        # 计算无目标的置信度损失
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')
        
        # 初始化响应掩码
        coord_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(0)
        coord_not_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(1)
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()
        
        # 遍历每个有目标的网格
        for i in range(0, bbox_target.size(0), B):
            # 获取当前网格的 B 个预测边界框
            pred = bbox_pred[i:i + B]
            
            # 将预测边界框转换为 (xmin, ymin, xmax, ymax) 格式
            pred_xyxy = Variable(torch.FloatTensor(pred.size()))
            pred_xyxy[:, :2] = pred[:, :2] / float(S) - 0.5 * pred[:, 2:4]  # 左上角
            pred_xyxy[:, 2:4] = pred[:, :2] / float(S) + 0.5 * pred[:, 2:4]  # 右下角
            
            # 获取当前网格的目标边界框
            target = bbox_target[i].view(-1, 5)
            
            # 将目标边界框转换为 (xmin, ymin, xmax, ymax) 格式
            target_xyxy = Variable(torch.FloatTensor(target.size()))
            target_xyxy[:, :2] = target[:, :2] / float(S) - 0.5 * target[:, 2:4]  # 左上角
            target_xyxy[:, 2:4] = target[:, :2] / float(S) + 0.5 * target[:, 2:4]  # 右下角
            
            # 计算预测边界框与目标边界框的 IoU
            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])
            max_iou, max_index = iou.max(0)  # 找出最大 IoU 及其索引
            max_index = max_index.data.cuda()
            
            # 标记负责预测目标的边界框
            coord_response_mask[i + max_index] = 1
            coord_not_response_mask[i + max_index] = 0
            
            # 将最大 IoU 作为置信度目标
            bbox_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = max_iou.data.cuda()
        
        bbox_target_iou = Variable(bbox_target_iou).cuda()
        
        # 提取负责预测目标的边界框
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)
        
        # 计算坐标损失 (中心点坐标)
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        
        # 计算宽高损失 (使用平方根以平衡大小目标)
        loss_wh = F.mse_loss(
            torch.sqrt(bbox_pred_response[:, 2:4]), 
            torch.sqrt(bbox_target_response[:, 2:4]),
            reduction='sum'
        )
        
        # 计算有目标的置信度损失
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')
        
        # 计算类别损失
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')
        
        # 计算总损失
        loss = (
            self.lambda_coord * (loss_xy + loss_wh) +  # 坐标损失
            loss_obj +                                # 有目标的置信度损失
            self.lambda_noobj * loss_noobj +           # 无目标的置信度损失
            loss_class                                # 类别损失
        )
        
        # 平均损失
        loss = loss / float(batch_size)
        
        return loss

# 测试代码
if __name__ == '__main__':
    loss = Detect_Loss(feature_size=7, num_bboxes=2, num_classes=20)
    pre = torch.randn(1, 7, 7, 30)  # 随机预测张量 [batch, S, S, B*5+C]
    tar = torch.randn(1, 7, 7, 30)  # 随机目标张量
    out = loss(pre.cuda(), tar.cuda())  # 计算损失
    print(out)