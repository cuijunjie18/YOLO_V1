## yolov1论文复现

### 背景

在进行多模态大模型进行目标检测前，先用经典的目标检测框架提高自己对目标检测流程、数据预处理、不同架构的优缺点及目标检测评估中各种概念的了解，夯实目标检测基础，加油吧!

### 目录结构说明

- datasets： 数据集存放位置，label为真实文件，原始图像为symbol link
- raw_paper： yolov1的原始论文
- utils： 存放常用功能脚本
- modules： 模型的组件
- train.py： 训练脚本
- main.ipynb： 整体流程的构思过程
- demo.ipynb： loss的调试

### 个人理解 + 收获

#### pytorch中mask的灵活运用

参考loss.py中的实现，发现许多操作都是mask提取实现的，如从一些连续的tensor中提取出某个部分，我们可以先生成对应的mask然后reshape即可.

注意下面两种方式

```py
# 注意要符合数据的要求，归一化后数值0~1
pre = torch.randn(1,30,7,7)  # 随机预测张量 [batch, S, S, B*5+C]
tar = torch.randn(1,30,7,7)  # 随机目标张量
mask1 = pre > 0
mask2 = tar > 0
pre[~mask1] = 0
tar[~mask2] = 0
print(pre)
# pre保持原形状
```

```py
# 设置有目标的mask
coord_mask = target[:,:,:,4] > 0
coord_mask = coord_mask.unsqueeze(-1).expand_as(target)
# 提取有目标的target
coord_target = target[coord_mask].reshape(-1, 5 * B + C)
```

#### 激活函数Relu的作用

搞了这么久终于理解了为什么要用Relu了，呜呜呜，原来目标检测任务后续的很多操作，如归一化，开根号等操作均要求值大于0,否则NAN！


### 参考文献

- yolo论文原文： https://arxiv.org/abs/1506.02640
- 知乎解读yolo论文： https://zhuanlan.zhihu.com/p/70387154
- github开源代码yolov1实现： https://github.com/yaoyi30/Pytorch_YOLOv1