import os
import shutil
import torch
from torchvision import transforms
from PIL import Image
from models.backbone import get_model

# ================= 配置区 =================
# 1. 模型路径
CHECKPOINT_PATH = "./outputs/checkpoints/best_model.pth"

# 2. 待处理的乱序图片文件夹 (输入)
INPUT_DIR = "./data/raw/new_batch_images"

# 3. 分类结果存放文件夹 (输出)
OUTPUT_DIR = "./data/sorted_result"

# 4. 类别列表 (必须与训练时的顺序完全一致！)
# 你可以查看 data/train 下的文件夹顺序，或者训练时的 log
CLASS_NAMES = [
    "01_overall", "02_carbon_slide", "03_fixing_bolts", "04_joint_bearing",
    "05_guide_rod", "06_lower_shunt", "07_head_shunt", "08_mid_shunt",
    "09_pan_head", "10_bracket", "11_camera"
]


# =========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在加载模型... (使用设备: {device})")

    # 1. 加载模型结构
    model = get_model(num_classes=len(CLASS_NAMES), pretrained=False, freeze_backbone=False)

    # 2. 加载训练好的权重
    # map_location确保在没有GPU的电脑上也能用CPU运行
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()  # 切换到推理模式

    # 3. 定义预处理 (只做缩放和归一化，不做随机增强)
    infer_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"📂 扫描文件夹: {INPUT_DIR}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    count = 0
    # 遍历所有图片
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if not file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue

            img_path = os.path.join(root, file)

            try:
                # 读取并处理图片
                image = Image.open(img_path).convert('RGB')
                input_tensor = infer_transform(image).unsqueeze(0).to(device)  # 增加 batch 维度

                # 推理
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, preds = torch.max(probs, 1)

                class_idx = preds.item()
                class_name = CLASS_NAMES[class_idx]
                conf_score = confidence.item()

                # 只有置信度大于 0.6 才分类，否则可以丢到 "unknown" 文件夹
                if conf_score > 0.6:
                    target_folder = os.path.join(OUTPUT_DIR, class_name)
                    os.makedirs(target_folder, exist_ok=True)

                    # 移动文件 (如果想保留原图用 shutil.copy)
                    shutil.move(img_path, os.path.join(target_folder, file))
                    print(f"✅ [{class_name}] (conf: {conf_score:.2f}) -> {file}")
                    count += 1
                else:
                    print(f"⚠️ [跳过] 置信度过低 ({conf_score:.2f}) -> {file}")

            except Exception as e:
                print(f"❌ 处理出错 {file}: {e}")

    print(f"\n🎉 处理完成！共分类归档 {count} 张图片。")
    print(f"结果保存在: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()