import torch

def nms(boxes, scores, threshold):
    """
    非极大值抑制(Non-Maximum Suppression)实现
    参数:
        boxes: 检测框坐标 [n, 4] 格式为 (x1, y1, x2, y2)
        scores: 每个检测框的置信度分数 [n]
        threshold: IoU重叠阈值，高于此阈值的框将被抑制
    
    返回:
        保留框的索引列表 (LongTensor)
    """
    # 提取所有框的坐标
    x1 = boxes[:, 0]  # 左上角x坐标 [n]
    y1 = boxes[:, 1]  # 左上角y坐标 [n]
    x2 = boxes[:, 2]  # 右下角x坐标 [n]
    y2 = boxes[:, 3]  # 右下角y坐标 [n]
    # 计算每个框的面积 (宽*高)
    areas = (x2 - x1) * (y2 - y1)  # [n]

    # 按置信度分数降序排序，获取排序后的索引
    _, ids_sorted = scores.sort(0, descending=True)  # [n]
    ids = []  # 用于存储最终保留框的索引
    
    # 循环处理，直到没有框剩余
    while ids_sorted.numel() > 0:
        # 获取当前最高分框的索引
        # 处理单个框的特殊情况
        i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
        ids.append(i)  # 将当前框添加到保留列表

        # 如果只剩一个框，直接退出循环
        if ids_sorted.numel() == 1:
            break

        # 计算当前框(i)与其他框(ids_sorted[1:])的交集区域
        # 交集区域的左上角坐标取较大值
        inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i])  # [m-1, ]
        inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i])  # [m-1, ]
        # 交集区域的右下角坐标取较小值
        inter_x2 = x2[ids_sorted[1:]].clamp(max=x2[i])  # [m-1, ]
        inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i])  # [m-1, ]
        # 计算交集区域的宽和高（确保非负）
        inter_w = (inter_x2 - inter_x1).clamp(min=0)  # [m-1, ]
        inter_h = (inter_y2 - inter_y1).clamp(min=0)  # [m-1, ]

        # 计算交集面积
        inters = inter_w * inter_h  # [m-1, ]
        # 计算并集面积 = 面积A + 面积B - 交集面积
        unions = areas[i] + areas[ids_sorted[1:]] - inters  # [m-1, ]
        # 计算IoU（交并比）
        ious = inters / unions  # [m-1, ]

        # 找出IoU低于阈值的框（需要保留的框）
        # nonzero()返回满足条件的索引，squeeze()去除多余维度
        ids_keep = (ious <= threshold).nonzero().squeeze()  # [m-1, ]
        # 如果没有框保留，退出循环
        if ids_keep.numel() == 0:
            break
        # 更新ids_sorted，保留低IoU的框（+1是因为ids_sorted[0]是当前框）
        ids_sorted = ids_sorted[ids_keep + 1]

    # 返回保留框的索引
    return torch.LongTensor(ids)


def decode(pred_tensor, grid_size, num_bboxes, conf_thresh, prob_thresh, nb_classes):
    """
    解码模型原始输出为可理解的检测结果
    
    参数:
        pred_tensor: 模型输出张量 [1, S, S, B*5+C]
        grid_size: 网格尺寸 S
        num_bboxes: 每个网格预测的边界框数量 B
        conf_thresh: 边界框置信度阈值
        prob_thresh: 最终得分阈值（框置信度*类别置信度）
        nb_classes: 类别数量 C
    
    返回:
        boxes: 检测框坐标 [n, 4] (归一化坐标)
        labels: 类别标签 [n]
        confidences: 框置信度 [n]
        class_scores: 类别置信度 [n]
    """
    # 解析参数
    S, B, C = grid_size, num_bboxes, nb_classes
    # 初始化结果列表
    boxes, labels, confidences, class_scores = [], [], [], []

    # 计算单元格大小（归一化坐标）
    cell_size = 1.0 / float(S)

    # 预处理张量：移除batch维度，转为CPU，分离梯度
    pred_tensor = pred_tensor.cpu().data.squeeze(0)

    # 提取所有边界框的置信度
    pred_tensor_conf_list = []
    for b in range(B):
        # 每个框的置信度在第4个位置 (x, y, w, h, conf)
        conf = pred_tensor[:, :, 5 * b + 4].unsqueeze(2)  # [S, S, 1]
        pred_tensor_conf_list.append(conf)
    
    # 合并所有框的置信度 [S, S, B]
    grid_ceil_conf = torch.cat(pred_tensor_conf_list, 2)
    # 获取每个网格中最可信的框及其置信度
    grid_ceil_conf, grid_ceil_index = grid_ceil_conf.max(2)  # [S, S], [S, S]
    
    # 处理类别部分 (最后C个通道)
    class_conf, class_index = pred_tensor[:, :, 5 * B:].max(2)  # [S, S], [S, S]
    # 应用类别置信度阈值
    class_conf[class_conf <= conf_thresh] = 0
    # 计算最终得分 = 框置信度 * 类别置信度
    class_prob = class_conf * grid_ceil_conf  # [S, S]

    # 遍历所有网格单元
    for i in range(S):  # x方向
        for j in range(S):  # y方向
            # 跳过得分低于阈值的网格
            if float(class_prob[j, i]) < prob_thresh:
                continue
                
            # 获取当前网格最佳框的参数
            box_index = grid_ceil_index[j, i]
            box = pred_tensor[j, i, 5 * box_index: 5 * box_index + 4]
            
            # 计算网格起始位置（归一化坐标）
            xy_start_pos = torch.FloatTensor([i, j]) * cell_size
            # 计算框中心点坐标（归一化）
            xy_normalized = box[:2] * cell_size + xy_start_pos
            # 获取框的宽高（归一化）
            wh_normalized = box[2:]
            
            # 转换为中心点坐标 -> 角点坐标 (xyxy格式)
            box_xyxy = torch.FloatTensor(4)
            box_xyxy[:2] = xy_normalized - 0.5 * wh_normalized  # 左上角
            box_xyxy[2:] = xy_normalized + 0.5 * wh_normalized  # 右下角

            # 添加到结果列表
            boxes.append(box_xyxy)
            labels.append(class_index[j, i])
            confidences.append(grid_ceil_conf[j, i])
            class_scores.append(class_conf[j, i])

    # 处理结果：转换为张量
    if len(boxes) > 0:
        boxes = torch.stack(boxes, 0)  # [n, 4]
        labels = torch.stack(labels, 0)  # [n]
        confidences = torch.stack(confidences, 0)  # [n]
        class_scores = torch.stack(class_scores, 0)  # [n]
    else:  # 没有检测到目标时返回空张量
        boxes = torch.FloatTensor(0, 4)
        labels = torch.LongTensor(0)
        confidences = torch.FloatTensor(0)
        class_scores = torch.FloatTensor(0)

    return boxes, labels, confidences, class_scores


def postprocess(output, width, height, VOC_CLASSES, grid_size, num_bboxes, 
                conf_thresh, prob_thresh, nms_thresh, nb_classes):
    """
    后处理主函数：整合解码和NMS，输出最终检测结果
    
    参数:
        output: 模型原始输出
        width, height: 原始图像尺寸
        VOC_CLASSES: 类别名称列表
        ... 其他参数同decode函数
        nms_thresh: NMS的IoU阈值
    
    返回:
        boxes: 检测框列表，格式为 [((x1,y1), (x2,y2)), ...]
        labels: 类别名称列表
        probs: 最终置信度列表
    """
    boxes, labels, probs = [], [], []  # 最终输出结果

    # 步骤1: 解码模型输出
    boxes_list, labels_list, confidences_list, class_scores_list = decode(
        output, grid_size, num_bboxes, conf_thresh, prob_thresh, nb_classes
    )

    # 如果有检测结果
    if boxes_list.shape[0] != 0:
        boxes_nms, labels_nms, probs_nms = [], [], []  # 存储NMS后的结果
        
        # 步骤2: 按类别进行非极大值抑制(NMS)
        for class_label in range(len(VOC_CLASSES)):
            # 获取当前类别的所有检测框
            ids = (labels_list == class_label)
            if torch.sum(ids) == 0:  # 如果没有当前类别的框，跳过
                continue

            # 提取当前类别的数据
            boxes_list_current_cls = boxes_list[ids]
            labels_list_current_cls = labels_list[ids]
            confidences_list_current_cls = confidences_list[ids]
            class_scores_list_current_cls = class_scores_list[ids]

            # 执行非极大值抑制
            ids_postprocess = nms(
                boxes_list_current_cls, 
                confidences_list_current_cls, 
                nms_thresh
            )

            # 收集NMS后的结果
            boxes_nms.append(boxes_list_current_cls[ids_postprocess])
            labels_nms.append(labels_list_current_cls[ids_postprocess])
            # 最终置信度 = 框置信度 * 类别置信度
            probs_nms.append(
                confidences_list_current_cls[ids_postprocess] * 
                class_scores_list_current_cls[ids_postprocess]
            )

        # 合并所有类别的结果
        boxes_nms = torch.cat(boxes_nms, 0)  # [n, 4]
        labels_nms = torch.cat(labels_nms, 0)  # [n]
        probs_nms = torch.cat(probs_nms, 0)    # [n]

        # 步骤3: 处理每个检测结果
        for box, label, prob in zip(boxes_nms, labels_nms, probs_nms):
            # 将归一化坐标转换为实际图像坐标
            x1, x2 = width * box[0], width * box[2]  # 反归一化x坐标
            y1, y2 = height * box[1], height * box[3]  # 反归一化y坐标
            boxes.append(((x1, y1), (x2, y2)))  # 存储为角点格式

            # 将类别索引转换为名称
            label_idx = int(label)
            class_name = VOC_CLASSES[label_idx]
            labels.append(class_name)

            # 转换概率为Python float
            prob = float(prob)
            probs.append(prob)

    # 返回最终检测结果
    return boxes, labels, probs