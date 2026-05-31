import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 设置中文字体（避免图表中文乱码）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ========== 配置项 ==========
MODEL_PATH = "models/plant_cls_model.pth"  # 训练好的模型权重
TEST_ROOT = "data/test/"  # 测试集根目录
VIS_SAVE_DIR = "visual_results"  # 可视化结果保存目录
CLASSES = ["八角金盘", "杜鹃", "广玉兰", "桂叶", "海桐",
           "木槿", "石楠", "梧桐", "银杏"]  # 中文类别名（便于可视化）
CLASSES_EN = ["BaJiaoJinPan", "DuJuan", "GuangYuLan", "GuiYe", "HaiTong",
              "MuJin", "ShiNan", "WuTong", "YinXing"]  # 英文文件夹名
IMG_SIZE = 224
DEVICE = torch.device("cpu")

# 创建可视化保存目录
os.makedirs(VIS_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(VIS_SAVE_DIR, "single_img"), exist_ok=True)  # 单张图像可视化
os.makedirs(os.path.join(VIS_SAVE_DIR, "charts"), exist_ok=True)  # 统计图表


# ========== ECA注意力类 ==========
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        kernel_size = int(abs((np.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# ========== 分类模型 ==========
class LightWeightClsModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ECA(16),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ECA(32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ECA(64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ECA(128),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ECA(256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


# ========== 加载模型 ==========
def load_model():
    model = LightWeightClsModel(num_classes=9).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    return model


# ========== 图像预处理 ==========
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ========== 1. 单张图像可视化（标注预测结果） ==========
def visualize_single_image(img_path, true_cls, pred_cls, pred_prob, save_path):
    """
    可视化单张图像，标注真实类别、预测类别、置信度
    :param img_path: 原始图像路径
    :param true_cls: 真实类别（中文）
    :param pred_cls: 预测类别（中文）
    :param pred_prob: 置信度
    :param save_path: 可视化结果保存路径
    """
    try:
        # 加载图像
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 设置字体（优先系统字体，无则用默认）
        try:
            font = ImageFont.truetype("simhei.ttf", 20)  # 黑体
        except:
            font = ImageFont.load_default(size=20)

        # 标注文本（区分正确/错误）
        if pred_cls == true_cls:
            text_color = (0, 255, 0)  # 绿色：正确
            status = "正确"
        else:
            text_color = (255, 0, 0)  # 红色：错误
            status = "错误"

        # 绘制文本
        text = f"真实：{true_cls}\n预测：{pred_cls}\n置信度：{pred_prob}%\n状态：{status}"
        y0 = 10
        for line in text.split("\n"):
            draw.text((10, y0), line, fill=text_color, font=font)
            y0 += 30

        # 保存可视化图像
        img.save(save_path)
        print(f"📸 单张可视化结果已保存：{save_path}")

    except Exception as e:
        print(f"⚠️  可视化单张图像失败 {img_path}：{str(e)}")


# ========== 2. 统计图表可视化 ==========
def visualize_stats(all_true_cls, all_pred_cls, class_stats):
    """
    生成统计图表：
    1. 每类准确率柱状图
    2. 混淆矩阵热力图
    3. 分类报告（精确率、召回率、F1）
    """
    # ========== 2.1 每类准确率柱状图 ==========
    plt.figure(figsize=(12, 6))
    cls_names = list(class_stats.keys())
    cls_accs = [
        round(class_stats[cls]["correct"] / class_stats[cls]["total"] * 100, 2) if class_stats[cls]["total"] > 0 else 0
        for cls in cls_names]

    # 绘制柱状图（按准确率排序）
    sorted_idx = np.argsort(cls_accs)[::-1]
    sorted_cls = [cls_names[i] for i in sorted_idx]
    sorted_accs = [cls_accs[i] for i in sorted_idx]

    bars = plt.bar(sorted_cls, sorted_accs,
                   color=["green" if acc >= 90 else "orange" if acc >= 80 else "red" for acc in sorted_accs])
    plt.title("9类植物分类 - 每类准确率", fontsize=14)
    plt.xlabel("植物类别", fontsize=12)
    plt.ylabel("准确率（%）", fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")

    # 在柱子上标注数值
    for bar, acc in zip(bars, sorted_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{acc}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_SAVE_DIR, "charts", "class_accuracy.png"), dpi=150)
    plt.close()
    print(f"📊 每类准确率图表已保存：class_accuracy.png")

    # ========== 2.2 混淆矩阵热力图 ==========
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_true_cls, all_pred_cls, labels=CLASSES)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("9类植物分类 - 混淆矩阵", fontsize=14)
    plt.xlabel("预测类别", fontsize=12)
    plt.ylabel("真实类别", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_SAVE_DIR, "charts", "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"📊 混淆矩阵图表已保存：confusion_matrix.png")

    # ========== 2.3 分类报告（保存为文本） ==========
    report = classification_report(all_true_cls, all_pred_cls, target_names=CLASSES, output_dict=True)
    report_text = classification_report(all_true_cls, all_pred_cls, target_names=CLASSES)

    # 保存分类报告到文本文件
    with open(os.path.join(VIS_SAVE_DIR, "charts", "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("9类植物分类 - 分类报告（精确率/召回率/F1）\n")
        f.write("=" * 80 + "\n")
        f.write(report_text)

    # 绘制F1分数柱状图
    plt.figure(figsize=(12, 6))
    f1_scores = [report[cls]["f1-score"] for cls in CLASSES]
    bars = plt.bar(CLASSES, f1_scores,
                   color=["green" if f1 >= 0.9 else "orange" if f1 >= 0.8 else "red" for f1 in f1_scores])
    plt.title("9类植物分类 - F1分数", fontsize=14)
    plt.xlabel("植物类别", fontsize=12)
    plt.ylabel("F1分数", fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    # 标注数值
    for bar, f1 in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{f1:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_SAVE_DIR, "charts", "f1_score.png"), dpi=150)
    plt.close()
    print(f"📊 F1分数图表已保存：f1_score.png")
    print(f"📝 分类报告已保存：classification_report.txt")


# ========== 3. 错误案例可视化汇总 ==========
def visualize_error_cases(error_cases):
    """生成错误案例汇总图（拼接多张错分图像）"""
    if not error_cases:
        print("✅ 无错误案例，跳过错误可视化")
        return

    # 最多显示20个错误案例（避免图像过大）
    error_cases = error_cases[:20]
    cols = 4  # 列数
    rows = (len(error_cases) + cols - 1) // cols  # 行数

    plt.figure(figsize=(cols * 5, rows * 6))
    for idx, case in enumerate(error_cases):
        # 加载图像
        img = Image.open(case["img_path"]).convert("RGB")
        img = img.resize((200, 200))  # 统一尺寸

        # 绘制子图
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"真实：{case['true_cls']}\n预测：{case['pred_cls']}\n置信度：{case['pred_prob']}%",
                  fontsize=10, color="red")

    plt.suptitle("9类植物分类 - 错误案例汇总", fontsize=16, color="red")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_SAVE_DIR, "error_cases.png"), dpi=150)
    plt.close()
    print(f"❌ 错误案例汇总图已保存：error_cases.png")


# ========== 批量预测+可视化主函数 ==========
def batch_predict_with_visual():
    # 1. 初始化模型和预处理
    model = load_model()
    transform = get_transform()

    # 2. 统计变量初始化
    total_count = 0
    correct_count = 0
    error_cases = []
    class_stats = {cls: {"total": 0, "correct": 0} for cls in CLASSES}
    all_true_cls = []  # 所有真实类别（用于统计）
    all_pred_cls = []  # 所有预测类别（用于统计）

    # 3. 遍历测试集
    print(f"开始批量预测+可视化：{TEST_ROOT}")
    print("=" * 80)

    for cls_en, cls_cn in zip(CLASSES_EN, CLASSES):
        cls_dir = os.path.join(TEST_ROOT, cls_en)
        if not os.path.exists(cls_dir):
            print(f"⚠️  类别文件夹不存在：{cls_dir}，跳过")
            continue

        img_paths = glob.glob(os.path.join(cls_dir, "*.[jp][pn]g")) + glob.glob(os.path.join(cls_dir, "*.jpeg"))
        if not img_paths:
            print(f"⚠️  类别 {cls_cn} 下无图像，跳过")
            continue

        for img_path in img_paths:
            total_count += 1
            class_stats[cls_cn]["total"] += 1

            # 预测单张图像
            pred_cls_en, pred_prob = predict_single_image(model, img_path, transform)
            pred_cls_cn = CLASSES[CLASSES_EN.index(pred_cls_en)] if pred_cls_en else "未知"

            # 记录真实/预测类别（用于统计）
            all_true_cls.append(cls_cn)
            all_pred_cls.append(pred_cls_cn)

            # 统计结果
            if pred_cls_cn == cls_cn:
                correct_count += 1
                class_stats[cls_cn]["correct"] += 1
                print(f"✅ {img_path} | 真实：{cls_cn} | 预测：{pred_cls_cn} | 置信度：{pred_prob}%")
            else:
                error_cases.append({
                    "img_path": img_path,
                    "true_cls": cls_cn,
                    "pred_cls": pred_cls_cn,
                    "pred_prob": pred_prob
                })
                print(f"❌ {img_path} | 真实：{cls_cn} | 预测：{pred_cls_cn} | 置信度：{pred_prob}%")

            # 单张图像可视化（保存到visual_results/single_img）
            img_name = os.path.basename(img_path)
            vis_save_path = os.path.join(VIS_SAVE_DIR, "single_img", f"vis_{img_name}")
            visualize_single_image(img_path, cls_cn, pred_cls_cn, pred_prob, vis_save_path)

    # 4. 输出文本统计结果
    print("=" * 80)
    print("📊 批量预测统计结果")
    print(f"总图像数：{total_count}")
    print(f"正确数：{correct_count}")
    print(f"整体准确率：{round(correct_count / total_count * 100, 2)}%")
    print("\n📈 每类准确率：")
    for cls_name, stats in class_stats.items():
        if stats["total"] == 0:
            acc = 0.0
        else:
            acc = round(stats["correct"] / stats["total"] * 100, 2)
        print(f"  {cls_name}：{stats['correct']}/{stats['total']} | {acc}%")

    # 5. 生成可视化图表
    print("\n🎨 开始生成可视化图表...")
    visualize_stats(all_true_cls, all_pred_cls, class_stats)
    visualize_error_cases(error_cases)

    # 6. 最终提示
    print("\n🎉 所有可视化结果已保存到：", VIS_SAVE_DIR)
    print("   - single_img：单张图像+预测结果标注")
    print("   - charts：统计图表（准确率/混淆矩阵/F1）")
    print("   - error_cases.png：错误案例汇总图")


# ========== 单张预测辅助函数 ==========
def predict_single_image(model, img_path, transform):
    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_cls_en = CLASSES_EN[pred_idx]
            pred_prob = round(torch.softmax(output, dim=1)[0][pred_idx].item() * 100, 2)

        return pred_cls_en, pred_prob

    except Exception as e:
        print(f"⚠️  处理图像失败 {img_path}：{str(e)}")
        return None, 0.0


# ========== 主函数 ==========
if __name__ == "__main__":
    # 安装依赖（若缺失）
    try:
        import seaborn
        import sklearn
    except:
        print("📦 安装可视化依赖库...")
        os.system("pip install seaborn scikit-learn -q")

    # 检查路径
    if not os.path.exists(TEST_ROOT):
        raise FileNotFoundError(f"测试集根目录不存在：{TEST_ROOT}")

    # 启动批量预测+可视化
    batch_predict_with_visual()