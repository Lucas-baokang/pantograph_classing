import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # 注册钩子，获取前向传播的激活值和反向传播的梯度
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx=None):
        # 1. 前向传播
        output = self.model(input_image)

        if class_idx is None:
            class_idx = torch.argmax(output)

        # 2. 反向传播获取梯度
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # 3. 计算神经元重要性权重 (Global Average Pooling on gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 4. 加权求和特征图 (ReLU 激活)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)

        # 5. 归一化处理
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def show_cam_on_image(img_path, mask):
    # 1. 读取原始高清图片 (例如 2448x3264)
    # 使用 imdecode 以支持中文路径
    img_bytes = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"无法找到图片或图片格式不支持: {img_path}")

    # 获取原图的尺寸 (高度, 宽度)
    height, width, _ = img.shape

    # 转换原图格式以便后续叠加
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255

    # 2. 处理热力图
    # 将灰度蒙版转换为彩色热力图 (此时它还是小尺寸，如 224x224)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    # =========== 关键修复步骤 START ===========
    # 将热力图拉伸到原图的大小
    # 注意：cv2.resize 的参数顺序是 (宽度, 高度)，与 shape 相反
    heatmap = cv2.resize(heatmap, (width, height))
    # =========== 关键修复步骤 END =============

    # 3. 叠加
    # 现在两个数组的形状都是 (2448, 3264, 3)，可以相加了
    # 乘以一个系数 (如 0.5) 可以调整热力图的透明度
    heatmap_strength = 0.6
    cam = heatmap * heatmap_strength + img * (1 - heatmap_strength)

    # 重新归一化到 0-1 之间以便显示
    cam = cam / np.max(cam)

    # 4. 绘图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image (High Res)")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Heatmap Overlay")
    plt.imshow(cam)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- 使用示例 ---
# 1. 加载你的 ResNet-50 模型
model = models.resnet50(pretrained=True)
model.eval()

# 2. 指定目标卷积层 (对于 ResNet-50，通常是 layer4 的最后一层)
target_layer = model.layer4[-1]

# 3. 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = "./data/type_A/train/02_carbon_slide/867_MP1_IMG_6366.JPG"  # 替换为你的照片路径
img_pil = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img_pil).unsqueeze(0)

# 4. 运行可视化
cam_extractor = GradCAM(model, target_layer)
mask = cam_extractor.generate_heatmap(input_tensor)
show_cam_on_image(img_path, mask)