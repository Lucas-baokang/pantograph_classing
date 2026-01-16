import torch.nn as nn
from torchvision import models


def get_model(num_classes, pretrained=True, freeze_backbone=True):
    # 1. 解决警告：使用新的 weights 参数代替 pretrained
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    # 2. 冻结骨干网络参数（只训练分类头）
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 3. 替换最后的全连接层 (fc)
    # ResNet50 的 fc 输入特征数是 2048
    num_ftrs = model.fc.in_features

    # 这里用到了 nn，所以必须在文件开头 import torch.nn as nn
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model