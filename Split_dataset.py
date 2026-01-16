import os
import shutil
import random
import glob
from tqdm import tqdm  # 如果没装 tqdm 可以去掉，或者 pip install tqdm

# ================= 配置区 =================
# 刚刚整理好的总数据目录
SOURCE_DIR = "./data/type_C/all_images_organized"

# 目标根目录
DATA_ROOT = "./data/type_C"

# 验证集比例 (0.2 = 20%)
VAL_RATIO = 0.2

# 设置随机种子，保证每次切分结果一致（可复现）
RANDOM_SEED = 42


# =========================================

def split_train_val():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 错误：找不到源目录 {SOURCE_DIR}")
        return

    print(f"🚀 开始切分数据集 (验证集比例: {VAL_RATIO})")

    # 设置随机种子
    random.seed(RANDOM_SEED)

    # 准备目标路径
    train_root = os.path.join(DATA_ROOT, "train")
    val_root = os.path.join(DATA_ROOT, "val")

    # 如果目标目录已存在，建议清理一下防止混淆（这里我选择覆盖/追加模式，但在打印时会提示）
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

    # 获取所有类别文件夹
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    classes.sort()  # 排序保证顺序一致

    total_train = 0
    total_val = 0

    print("-" * 40)
    print(f"{'类别':<20} | {'总数':<6} | {'训练集':<6} | {'验证集':<6}")
    print("-" * 40)

    for class_name in classes:
        class_src_path = os.path.join(SOURCE_DIR, class_name)

        # 获取该类下所有图片
        images = glob.glob(os.path.join(class_src_path, "*.*"))
        # 过滤非图片文件
        images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # 随机打乱
        random.shuffle(images)

        # 计算切分索引
        count = len(images)
        split_idx = int(count * (1 - VAL_RATIO))

        # 训练集和验证集列表
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # 执行复制
        # 复制到 train
        dst_train_dir = os.path.join(train_root, class_name)
        os.makedirs(dst_train_dir, exist_ok=True)
        for img in train_imgs:
            shutil.copy2(img, os.path.join(dst_train_dir, os.path.basename(img)))

        # 复制到 val
        dst_val_dir = os.path.join(val_root, class_name)
        os.makedirs(dst_val_dir, exist_ok=True)
        for img in val_imgs:
            shutil.copy2(img, os.path.join(dst_val_dir, os.path.basename(img)))

        # 打印统计
        print(f"{class_name:<20} | {count:<6} | {len(train_imgs):<6} | {len(val_imgs):<6}")

        total_train += len(train_imgs)
        total_val += len(val_imgs)

    print("-" * 40)
    print(f"✅ 切分完成！")
    print(f"训练集总数: {total_train} (保存在 {train_root})")
    print(f"验证集总数: {total_val} (保存在 {val_root})")


if __name__ == "__main__":
    split_train_val()