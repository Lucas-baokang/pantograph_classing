import os
import copy
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json

# 引入我们在 models/backbone.py 定义的模型
from models.backbone import get_model


# ================= 配置区 =================
#  (配置路径现在通过命令行读取，这里留空即可)
# =========================================

# 【新增】补回这个读取 yaml 文件的辅助函数
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    # --- 修改开始: 使用命令行参数读取配置 ---
    parser = argparse.ArgumentParser(description='Train Pantograph Classifier')
    # 默认路径设为 type_B.yaml，方便你直接点运行
    parser.add_argument('--config', type=str, default='./configs/type_C.yaml', help='Path to config file')
    args = parser.parse_args()

    CONFIG_PATH = args.config

    # 检查配置文件是否存在，防止报错懵逼
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"❌ 找不到配置文件: {CONFIG_PATH}，请检查路径！")

    cfg = load_config(CONFIG_PATH)  # 现在这里可以正常工作了
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动训练，读取配置: {CONFIG_PATH}")
    print(f"📂 数据集路径: {cfg['data']['root_dir']}")
    print(f"💾 模型将保存至: ./outputs/{cfg['project_name']}")

    # 2. 数据预处理与增强
    # 训练集：增加随机扰动，防止过拟合
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(cfg['data']['input_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((cfg['data']['input_size'], cfg['data']['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 3. 加载数据集 (使用 ImageFolder 自动读取文件夹分类)
    data_dir = cfg['data']['root_dir']

    # 增加路径检查
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ 找不到数据目录: {data_dir}，请确认 type_B 数据是否已生成！")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x],
                                 batch_size=cfg['data']['batch_size'],
                                 shuffle=(x == 'train'),  # 训练集打乱，验证集不需要
                                 num_workers=cfg['data']['num_workers'])
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"📊 数据概览: 训练集 {dataset_sizes['train']} 张 | 验证集 {dataset_sizes['val']} 张")
    print(f"🏷️ 检测到 {len(class_names)} 个类别: {class_names}")

    # 4. 初始化模型
    # 注意：这里读取的是 len(class_names)，所以 yaml 里的 num_classes 写错了也不影响，以文件夹实际数量为准
    model = get_model(num_classes=len(class_names), pretrained=cfg['model']['pretrained'])
    model = model.to(device)

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg['train']['learning_rate'])

    # 6. 训练循环
    num_epochs = cfg['train']['epochs']
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 保存路径根据 project_name 动态生成
    save_dir = os.path.join("./outputs", cfg['project_name'])
    os.makedirs(save_dir, exist_ok=True)

    # 保存 json 标签
    with open(os.path.join(save_dir, 'classes.json'), 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"📝 类别映射表已保存至: {os.path.join(save_dir, 'classes.json')}")

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 每个 epoch 都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 进度条
            pbar = tqdm(dataloaders[phase], desc=f"{phase} Phase", unit="batch")

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播 (只在训练阶段)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 更新进度条显示的当前 Loss
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度复制最优模型 (基于验证集准确率)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                print(f"✨ 新的最优模型已保存 (Acc: {best_acc:.4f})")

    print(f'\n🏁 训练完成。最优验证集准确率: {best_acc:.4f}')
    print(f"💾 最终模型位于: {os.path.abspath(save_dir)}")


if __name__ == '__main__':
    main()