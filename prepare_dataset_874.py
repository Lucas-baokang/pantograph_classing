import os
import shutil
from pathlib import Path

# ================= 配置区 =================
# 原始数据的根目录 (请确认这个路径是否正确)
SOURCE_ROOT = "./data/raw/1期受电弓照片"

# 整理后的输出目录 (脚本会自动创建这个目录)
OUTPUT_ROOT = "./data/type_C/all_images_organized"

# 类别映射表 (关键词 -> 英文标准名)
# ⚠️注意顺序：长词在前，短词在后，防止误判 (例如 "弓头支座" 包含 "弓头")
CLASS_MAP = {
    # 1-8类保持不变，或者也简化一下
    "1. 总体": "01_overall",
    "1.总体": "01_overall",  # 防空格丢失

    "2. 碳滑条": "02_carbon_slide",
    "2.碳滑条": "02_carbon_slide",

    "3. 碳棒": "03_fixing_bolts",
    "3.碳棒": "03_fixing_bolts",

    "4. 拉杆": "04_rod_bearing",
    "4.拉杆": "04_rod_bearing",

    "5. 平衡杆": "05_balance_rod",
    "5.平衡杆": "05_balance_rod",

    # 6,7,8 因为名字很像，还是得保留较长的前缀，但我们要加上无空格版
    "6. 下支架": "06_lower_shunt",
    "6.下支架": "06_lower_shunt",

    "7. 弓头支座": "07_head_shunt",
    "7.弓头支座": "07_head_shunt",

    "8. 下支架与上": "08_mid_shunt",
    "8.下支架与上": "08_mid_shunt",

    # === 这里是修复 9 和 10 的关键 ===
    # 直接匹配核心名词，放弃前面的数字和"受电弓"前缀，这样最稳
    "9.弓头": "09_head",  # 只要文件夹里有"托架"二字，就归为09
    "托架": "10_bracket",  # 只要有"钢丝绳"二字，就归为10


}

# =========================================

def flatten_and_organize():
    if not os.path.exists(SOURCE_ROOT):
        print(f"❌ 错误：找不到源目录 {SOURCE_ROOT}，请检查路径配置。")
        return

    print(f"🚀 开始扫描目录: {SOURCE_ROOT}")
    print(f"📂 目标输出目录: {OUTPUT_ROOT}")

    # 计数器
    count_dict = {k: 0 for k in CLASS_MAP.values()}
    total_copied = 0

    # 遍历源目录的所有子文件夹
    for root, dirs, files in os.walk(SOURCE_ROOT):
        folder_name = os.path.basename(root)

        # 1. 判断当前文件夹是否是我们需要的“部件文件夹”
        target_class = None
        for key, value in CLASS_MAP.items():
            if key in folder_name:
                target_class = value
                break

        # 如果不是部件文件夹，跳过
        if not target_class:
            continue

        # 2. 获取该图片的“列车号”和“MP号”上下文信息
        # 假设路径结构是: .../874受电弓照片/mp1/1. 总体照片/image.jpg
        # root 是 .../874受电弓照片/mp1/1. 总体照片
        try:
            path_parts = Path(root).parts
            # 倒数第2级应该是 mp1 或 mp2
            mp_name = path_parts[-2]
            # 倒数第3级应该是 列车号文件夹
            train_name = path_parts[-3]
        except IndexError:
            mp_name = "unknown_mp"
            train_name = "unknown_train"

        # 3. 处理该文件夹下的所有图片
        target_dir = os.path.join(OUTPUT_ROOT, target_class)
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                src_path = os.path.join(root, file)

                # 4. 生成新文件名：列车号_MP号_原文件名
                # 例如：874受电弓照片_mp1_DSC0001.jpg
                new_filename = f"{train_name}_{mp_name}_{file}"

                # 清洗文件名中可能存在的非法字符（可选）
                new_filename = new_filename.replace(" ", "_")

                dst_path = os.path.join(target_dir, new_filename)

                # 5. 复制文件
                shutil.copy2(src_path, dst_path)

                count_dict[target_class] += 1
                total_copied += 1

    # ================= 打印报告 =================
    print("\n" + "=" * 30)
    print("✅ 整理完成！")
    print("=" * 30)
    for cls_name, count in sorted(count_dict.items()):
        print(f"  - {cls_name}: {count} 张")
    print("-" * 30)
    print(f"总计提取: {total_copied} 张图片")
    print(f"文件已保存在: {os.path.abspath(OUTPUT_ROOT)}")


if __name__ == "__main__":
    flatten_and_organize()