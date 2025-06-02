import torch
from torch import nn
import torch.nn.functional as F

class YoloLoss(nn.Module):
    """定义一个为yolov1的损失函数"""

    def __init__(self,feature_size=7, num_bboxes=2, num_classes=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self,bbox1,bbox2):
        """"
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


    def forward(self,pred:torch.Tensor,target:torch.Tensor):
        """
        计算 YOLOv1 损失
        
        参数:
        - pred_tensor: 模型预测的输出张量，形状为 [batch_size, S, S, B*5 + C]
        - target_tensor: 目标标签张量，形状与 pred_tensor 相同
        
        返回:
        - loss: 计算得到的损失值
        """
        # target/pred = (N,C,H,W) -> (N,H,W,C)
        device = target.device
        target = target.permute(0,2,3,1)
        pred = pred.permute(0,2,3,1)
        batch_size = pred.shape[0]

        # breakpoint()

        # 设置临时参数，减少重复self引用
        S = self.S
        B = self.B
        C = self.C
        grid_size = 1.0 / S # 归一化的网格大小

        # 设置有目标的mask和没目标的mask
        coord_mask = target[:,:,:,4] > 0
        noobj_mask = target[:,:,:,4] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)

        # 提取有目标的pred和没目标的pred
        coord_pred = pred[coord_mask].reshape(-1, 5 * B + C)
        noobj_pred = pred[noobj_mask].reshape(-1, 5 * B + C)

        # 提取有目标的target和没目标的target
        coord_target = target[coord_mask].reshape(-1, 5 * B + C)
        noobj_target = target[noobj_mask].reshape(-1, 5 * B + C)

        # 提取bbox与class
        bbox_pred = coord_pred[:,:5 * B].reshape(-1, 5)
        class_pred = coord_pred[:,5 * B:]
        bbox_target = coord_target[:,:5 * B].reshape(-1, 5)
        class_target = coord_target[:,5 * B:]

        # 处理无目标位置的置信度损失
        noobj_conf_mask = torch.BoolTensor(noobj_pred.shape).fill_(0).to(device)
        for b in range(B):
            noobj_conf_mask[:,4 + b * 5] = 1 # 设置提取出置信度的位置
        noobj_conf_pred = noobj_pred[noobj_conf_mask]
        noonj_conf_target = noobj_target[noobj_conf_mask]

        # 计算noobj_loss_conf
        loss_conf_noobj = F.mse_loss(noobj_conf_pred,noonj_conf_target,reduction = 'sum')

        # 初始化响应掩码
        coord_response_mask = torch.BoolTensor(bbox_target.size()).fill_(0).to(device) # 响应初始化为0
        coord_not_response_mask = torch.BoolTensor(bbox_target.size()).fill_(1).to(device) # 非响应初始化为1
        bbox_target_iou = torch.zeros(bbox_target.size()).to(device)

        # 遍历每个目标网格
        for i in range(0,bbox_pred.shape[0],B):
            # 获取当前网格的 B 个预测边界框
            pred = bbox_pred[i:i + B] 

            # 将预测边界框转换为 (xmin, ymin, xmax, ymax) 格式
            pred_xyxy = torch.zeros((pred.shape[0],4)).to(device)
            pred_xyxy[:,:2] = pred[:,:2] * grid_size - pred[:,2:4] * 0.5    # 左上
            pred_xyxy[:,2:4] = pred[:,:2] * grid_size + pred[:,2:4] * 0.5   # 右下

            # 同样处理出目标坐标
            target = bbox_target[i].reshape(-1,5)
            target_xyxy = torch.zeros((target.shape[0],4)).to(device)
            target_xyxy[:,:2] = target[:,:2] * grid_size - target[:,2:4] * 0.5    # 左上
            target_xyxy[:,2:4] = target[:,:2] * grid_size + target[:,2:4] * 0.5   # 右下

            # 计算iou
            iou = self.compute_iou(pred_xyxy,target_xyxy)
            max_iou, max_index = iou.max(0)  # 找出最大 IoU 及其索引

            # 标记负责预测目标的边界框
            coord_response_mask[i + max_index] = 1
            coord_not_response_mask[i + max_index] = 0

            # 将最大 IoU 作为置信度目标
            bbox_target_iou[i + max_index,4] = max_iou.data

        # 提取负责预测目标的边界框
        bbox_pred_response = bbox_pred[coord_response_mask].reshape(-1, 5)
        bbox_target_response = bbox_target[coord_response_mask].reshape(-1, 5)
        target_iou = bbox_target_iou[coord_response_mask].reshape(-1, 5)

        # 计算坐标损失 (中心点坐标)
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        
        # 计算宽高损失
        loss_wh = F.mse_loss(
            torch.sqrt(bbox_pred_response[:,2:4]), 
            torch.sqrt(bbox_target_response[:,2:4]), 
            reduction = 'sum'
        )

        # 计算目标置信度损失
        loss_conf = F.mse_loss(bbox_pred_response[:,4], target_iou[:,4], reduction = 'sum')

        # 计算类别预测损失
        loss_class = F.mse_loss(class_pred,class_target,reduction = 'sum')

        loss = (self.lambda_coord * (loss_xy + loss_wh) + 
                self.lambda_noobj * loss_conf_noobj +
                loss_conf + loss_class)
        
        return loss / batch_size # 平均损失

if __name__ == "__main__":
    loss = YoloLoss(feature_size = 7, num_bboxes = 2, num_classes = 20)
    pre = torch.randn(1,30,7,7)  # 随机预测张量 [batch, S, S, B*5+C]
    tar = torch.randn(1,30,7,7)  # 随机目标张量
    out = loss(pre,tar)  # 计算损失
    print(out)