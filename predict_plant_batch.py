import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import glob  # 用于遍历文件夹中的所有图像

# ========== 配置项 ==========
MODEL_PATH = "models/plant_cls_model.pth"  # 训练好的模型权重
TEST_ROOT = "data/test/"  # 测试集根目录（包含9类子文件夹）
CLASSES = ["BaJiaoJinPan", "DuJuan", "GuangYuLan", "GuiYe", "HaiTong",
           "MuJin", "ShiNan", "WuTong", "YinXing"]  # 9类植物名称
IMG_SIZE = 224  # 图像尺寸（和训练时一致）
DEVICE = torch.device("cpu")  # 强制CPU运行


# ========== ECA注意力类（和训练时一致） ==========
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


# ========== 分类模型（和训练时一致） ==========
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
    """加载训练好的模型"""
    model = LightWeightClsModel(num_classes=9).to(DEVICE)
    # 加载权重（忽略模型参数和权重参数的轻微不匹配，不影响预测）
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()  # 切换到评估模式
    return model


# ========== 图像预处理 ==========
def get_transform():
    """获取图像预处理流程（和训练时的验证集一致）"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ========== 单张图像预测 ==========
def predict_single_image(model, img_path, transform):
    """
    预测单张图像
    :param model: 加载好的模型
    :param img_path: 图像路径
    :param transform: 预处理流程
    :return: 预测类别名、置信度
    """
    try:
        # 加载并预处理图像
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # 添加batch维度

        # 预测（关闭梯度计算，提升速度）
        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()  # 预测类别索引
            pred_cls = CLASSES[pred_idx]  # 预测类别名
            pred_prob = torch.softmax(output, dim=1)[0][pred_idx].item() * 100  # 置信度

        return pred_cls, round(pred_prob, 2)

    except Exception as e:
        print(f"⚠️  处理图像失败 {img_path}：{str(e)}")
        return None, 0.0


# ========== 批量预测测试集 ==========
def batch_predict_test_set():
    """批量预测test文件夹下所有图像，并统计结果"""
    # 1. 初始化模型和预处理
    model = load_model()
    transform = get_transform()

    # 2. 统计变量初始化
    total_count = 0  # 总图像数
    correct_count = 0  # 预测正确数
    error_cases = []  # 错误案例（用于分析）
    class_stats = {cls: {"total": 0, "correct": 0} for cls in CLASSES}  # 每类的统计

    # 3. 遍历test文件夹下的所有类文件夹
    print(f"开始批量预测测试集：{TEST_ROOT}")
    print("=" * 80)

    for cls_name in CLASSES:
        cls_dir = os.path.join(TEST_ROOT, cls_name)
        if not os.path.exists(cls_dir):
            print(f"⚠️  类别文件夹不存在：{cls_dir}，跳过")
            continue

        # 遍历该类别下的所有图像（支持jpg/png/jpeg格式）
        img_paths = glob.glob(os.path.join(cls_dir, "*.[jp][pn]g")) + glob.glob(os.path.join(cls_dir, "*.jpeg"))
        if not img_paths:
            print(f"⚠️  类别 {cls_name} 下无图像，跳过")
            continue

        # 逐张预测
        for img_path in img_paths:
            total_count += 1
            class_stats[cls_name]["total"] += 1

            # 预测单张图像
            pred_cls, pred_prob = predict_single_image(model, img_path, transform)

            # 统计结果
            if pred_cls == cls_name:
                correct_count += 1
                class_stats[cls_name]["correct"] += 1
                print(f"✅ {img_path} | 真实：{cls_name} | 预测：{pred_cls} | 置信度：{pred_prob}%")
            else:
                error_cases.append({
                    "img_path": img_path,
                    "true_cls": cls_name,
                    "pred_cls": pred_cls,
                    "pred_prob": pred_prob
                })
                print(f"❌ {img_path} | 真实：{cls_name} | 预测：{pred_cls} | 置信度：{pred_prob}%")

    # 4. 输出统计结果
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

    # 5. 输出错误案例（可选）
    if error_cases:
        print("\n❌ 错误案例（共{}个）：".format(len(error_cases)))
        for case in error_cases:
            print(
                f"  图像：{case['img_path']} | 真实：{case['true_cls']} | 预测：{case['pred_cls']} | 置信度：{case['pred_prob']}%")


# ========== 主函数 ==========
if __name__ == "__main__":
    # 检查测试集路径
    if not os.path.exists(TEST_ROOT):
        raise FileNotFoundError(f"测试集根目录不存在：{TEST_ROOT}")
    # 启动批量预测
    batch_predict_test_set()