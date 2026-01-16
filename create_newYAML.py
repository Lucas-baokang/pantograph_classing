import os
import yaml

# 1. 定义 YAML 内容
config_data = {
    "project_name": "pantograph_cls_v1",
    "model": {
        "name": "resnet50",
        "num_classes": 11,
        "pretrained": True
    },
    "data": {
        "root_dir": "./pantograph_cls/data", # 注意：这里要确保指向你实际的数据父目录
        "input_size": 224,
        "batch_size": 32,
        "num_workers": 4
    },
    "train": {
        "learning_rate": 0.001,
        "epochs": 20,
        "seed_size": 250
    },
    "active_learning": {
        "query_size": 100,
        "strategy": "entropy"
    }
}

# 2. 确保目标文件夹存在
target_dir = "./pantograph_cls/configs"  # 或者是 "./configs" 取决于你在哪个目录下运行
os.makedirs(target_dir, exist_ok=True)

# 3. 写入文件
target_file = os.path.join(target_dir, "resnet50.yaml")
with open(target_file, 'w', encoding='utf-8') as f:
    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"✅ 成功创建配置文件: {os.path.abspath(target_file)}")