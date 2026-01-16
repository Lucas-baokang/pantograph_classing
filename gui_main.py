import os
import json
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from PIL import Image

import torch
from torchvision import transforms

# 引入项目自带的模型结构
from models.backbone import get_model

# ================= 配置区 =================
# 1. 默认置信度阈值
CONFIDENCE_THRESHOLD = 0.6

# 2. 基础选项配置
TEAMS = ["日检一班", "日检二班", "日检三班", "日检四班"]
BASES = ["殷行基地", "浦江基地"]
POSITIONS = ["mp1", "mp2"]

# 3. 自动生成标准车号列表 (下拉菜单用)
# Type C: 801-828, Type B: 829-866, Type A: 867-890
TRAIN_ID_LIST = []
TRAIN_ID_LIST.extend([str(i) for i in range(801, 829)])  # 801-828
TRAIN_ID_LIST.extend([str(i) for i in range(829, 867)])  # 829-866
TRAIN_ID_LIST.extend([str(i) for i in range(867, 891)])  # 867-890


# 4. 车号与模型的映射逻辑
def get_model_type_by_train_id(train_id_str):
    try:
        tid = int(train_id_str)
        if 801 <= tid <= 828:
            return "type_C"
        elif 829 <= tid <= 866:
            return "type_B"
        elif 867 <= tid <= 890:
            return "type_A"
        else:
            return None
    except ValueError:
        return None


# =========================================

class PantographSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("受电弓零件智能分类系统 V3.0 (标准版)")
        self.root.geometry("650x600")
        self.root.resizable(False, False)

        # 变量绑定
        self.train_id_var = tk.StringVar()
        self.position_var = tk.StringVar(value=POSITIONS[0])
        self.team_var = tk.StringVar(value=TEAMS[0])
        self.base_var = tk.StringVar(value=BASES[0])
        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪")

        self.is_running = False

        self._init_ui()

    def _init_ui(self):
        # --- 基础信息区域 ---
        info_frame = ttk.LabelFrame(self.root, text="作业信息配置", padding=15)
        info_frame.pack(fill="x", padx=10, pady=10)

        # Grid 布局权重
        info_frame.columnconfigure(1, weight=1)
        info_frame.columnconfigure(3, weight=1)

        # 第一行：列车号 (改为 Combobox) & 受电弓位置
        ttk.Label(info_frame, text="列车号:").grid(row=0, column=0, padx=5, pady=8, sticky="e")
        # 【修改点1】改为下拉菜单，限制只能从列表中选择
        train_combo = ttk.Combobox(info_frame, textvariable=self.train_id_var, values=TRAIN_ID_LIST, width=18,
                                   state="readonly")
        train_combo.grid(row=0, column=1, padx=5, pady=8, sticky="w")
        if TRAIN_ID_LIST: train_combo.current(0)  # 默认选第一个

        ttk.Label(info_frame, text="受电弓位置:").grid(row=0, column=2, padx=5, pady=8, sticky="e")
        ttk.Combobox(info_frame, textvariable=self.position_var, values=POSITIONS, width=15, state="readonly").grid(
            row=0, column=3, padx=5, pady=8, sticky="w")

        # 第二行：工作班组 & 基地
        ttk.Label(info_frame, text="工作班组:").grid(row=1, column=0, padx=5, pady=8, sticky="e")
        ttk.Combobox(info_frame, textvariable=self.team_var, values=TEAMS, width=16, state="readonly").grid(row=1,
                                                                                                            column=1,
                                                                                                            padx=5,
                                                                                                            pady=8,
                                                                                                            sticky="w")

        ttk.Label(info_frame, text="作业基地:").grid(row=1, column=2, padx=5, pady=8, sticky="e")
        ttk.Combobox(info_frame, textvariable=self.base_var, values=BASES, width=15, state="readonly").grid(row=1,
                                                                                                            column=3,
                                                                                                            padx=5,
                                                                                                            pady=8,
                                                                                                            sticky="w")

        # --- 路径选择区域 ---
        path_frame = ttk.LabelFrame(self.root, text="文件路径", padding=15)
        path_frame.pack(fill="x", padx=10, pady=5)

        # 导入路径
        ttk.Label(path_frame, text="待处理图片文件夹:").grid(row=0, column=0, sticky="w")
        entry_in = ttk.Entry(path_frame, textvariable=self.input_path_var, width=55)
        entry_in.grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(path_frame, text="浏览...", command=lambda: self.select_dir(self.input_path_var)).grid(row=1,
                                                                                                          column=1,
                                                                                                          padx=5)

        # 导出路径
        ttk.Label(path_frame, text="分类结果保存位置:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        entry_out = ttk.Entry(path_frame, textvariable=self.output_path_var, width=55)
        entry_out.grid(row=3, column=0, padx=5, pady=5)
        ttk.Button(path_frame, text="浏览...", command=lambda: self.select_dir(self.output_path_var)).grid(row=3,
                                                                                                           column=1,
                                                                                                           padx=5)

        # --- 控制区域 ---
        ctrl_frame = ttk.Frame(self.root, padding=10)
        ctrl_frame.pack(fill="x", padx=10)

        self.btn_start = ttk.Button(ctrl_frame, text="开始智能识别与归档", command=self.start_processing)
        self.btn_start.pack(fill="x", ipady=8)

        # 进度条
        self.progress = ttk.Progressbar(ctrl_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill="x", pady=15)

        # 状态栏
        ttk.Label(ctrl_frame, textvariable=self.status_var, foreground="blue").pack(anchor="w")

        # --- 日志区域 ---
        log_frame = ttk.LabelFrame(self.root, text="运行日志", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_text = tk.Text(log_frame, height=8, state="disabled", font=("微软雅黑", 9))
        self.log_text.pack(fill="both", expand=True)

    def select_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def start_processing(self):
        if self.is_running:
            return

        train_id = self.train_id_var.get().strip()
        input_dir = self.input_path_var.get()
        output_dir = self.output_path_var.get()
        team_name = self.team_var.get()
        base_name = self.base_var.get()
        pos_name = self.position_var.get()

        if not train_id:
            messagebox.showwarning("提示", "请选择列车号！")
            return
        if not input_dir or not output_dir:
            messagebox.showwarning("提示", "请选择完整的输入和输出路径！")
            return

        model_type = get_model_type_by_train_id(train_id)

        if not model_type:
            # 理论上用了下拉菜单不会进这里，但为了保险还是留着
            messagebox.showerror("错误", f"无法匹配车型，请确认车号 {train_id} 是否正确。")
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.status_var.set(f"正在加载 {model_type} 模型 (适配车号 {train_id})...")
        self.progress['value'] = 0

        thread = threading.Thread(target=self.run_inference_task,
                                  args=(model_type, train_id, input_dir, output_dir, team_name, base_name, pos_name))
        thread.daemon = True
        thread.start()

    def run_inference_task(self, model_type, train_id, input_dir, output_dir, team_name, base_name, pos_name):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # --- 路径构建 ---
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_path, "outputs", model_type, "best_model.pth")
            json_path = os.path.join(base_path, "outputs", model_type, "classes.json")

            if not os.path.exists(model_path) or not os.path.exists(json_path):
                raise FileNotFoundError(f"找不到 {model_type} 资源文件！\n请检查 outputs 目录。")

            # --- 加载资源 ---
            with open(json_path, 'r', encoding='utf-8') as f:
                class_names = json.load(f)

            self.root.after(0, self.log, f"加载模型: {model_type}")

            model = get_model(num_classes=len(class_names), pretrained=False, freeze_backbone=False)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()

            # --- 预处理 ---
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # --- 扫描文件 ---
            image_files = []
            for root, dirs, files in os.walk(input_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_files.append(os.path.join(root, f))

            total_files = len(image_files)
            if total_files == 0:
                self.root.after(0, messagebox.showinfo, "提示", "目录下没有图片！")
                return

            # --- 【修改点2】准备输出目录结构 ---
            today_str = datetime.now().strftime("%Y%m%d")

            # 1. 创建作业根目录 (包含日期、班组、基地)
            job_folder_name = f"{train_id}_{today_str}_{team_name}_{base_name}"
            job_root = os.path.join(output_dir, job_folder_name)

            # 2. 在作业目录下，创建 mp1 或 mp2 子目录 (根据 GUI 选择)
            # 最终路径示例: outputs/874_.../mp1/
            position_root = os.path.join(job_root, pos_name)

            # 未识别目录也放在 mp1/mp2 下
            unknown_dir = os.path.join(position_root, "Unknown_无法识别")
            os.makedirs(unknown_dir, exist_ok=True)

            self.root.after(0, self.log, f"输出目录: {position_root}")

            success_count = 0
            fail_count = 0
            report_lines = []

            report_lines.append(f"=== 作业报告 ===")
            report_lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"列车: {train_id} ({model_type}) | 位置: {pos_name}")
            report_lines.append(f"班组: {team_name} | 基地: {base_name}")
            report_lines.append("-" * 30)

            # --- 推理循环 ---
            for i, img_path in enumerate(image_files):
                original_filename = os.path.basename(img_path)

                try:
                    img = Image.open(img_path).convert('RGB')
                    input_tensor = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        conf, preds = torch.max(probs, 1)

                    conf_val = conf.item()
                    pred_idx = preds.item()

                    # 命名格式: 列车号_日期_班组_基地_原文件名
                    new_filename = f"{train_id}_{today_str}_{team_name}_{base_name}_{original_filename}"

                    if conf_val >= CONFIDENCE_THRESHOLD:
                        class_name = class_names[pred_idx]

                        # 【核心修改】将分类文件夹创建在 position_root (即 mp1/mp2) 下
                        target_folder = os.path.join(position_root, class_name)
                        os.makedirs(target_folder, exist_ok=True)

                        shutil.copy2(img_path, os.path.join(target_folder, new_filename))
                        success_count += 1
                    else:
                        # 放入 Unknown
                        shutil.copy2(img_path, os.path.join(unknown_dir, new_filename))
                        report_lines.append(f"[无效] {original_filename} (置信度: {conf_val:.2f})")
                        fail_count += 1

                except Exception as e:
                    report_lines.append(f"[错误] {original_filename}: {str(e)}")
                    fail_count += 1

                progress_val = (i + 1) / total_files * 100
                self.root.after(0, self.update_progress, progress_val, f"处理中: {i + 1}/{total_files}")

            # --- 结束工作 ---
            # 报告保存在 mp1/mp2 目录下，或者上一级都可以，这里保存在 mp1/mp2 下最清晰
            report_path = os.path.join(position_root, "处理报告.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"处理结果统计:\n成功归档: {success_count}\n失败/无效: {fail_count}\n\n")
                f.write("\n".join(report_lines))

            self.root.after(0, self.log, f"完成！成功 {success_count}，异常 {fail_count}")
            self.root.after(0, messagebox.showinfo, "作业完成", f"归档完毕！\n请查看: {position_root}")

            # 打开文件夹 (打开到 mp1 或 mp2 这一层)
            os.startfile(position_root)

        except Exception as e:
            self.root.after(0, messagebox.showerror, "运行错误", str(e))
            self.root.after(0, self.log, f"CRITICAL ERROR: {str(e)}")

        finally:
            self.is_running = False
            self.root.after(0, self.reset_ui)

    def update_progress(self, val, msg):
        self.progress['value'] = val
        self.status_var.set(msg)

    def reset_ui(self):
        self.btn_start.config(state="normal")
        self.status_var.set("就绪")


if __name__ == "__main__":
    root = tk.Tk()
    app = PantographSorterApp(root)
    root.mainloop()