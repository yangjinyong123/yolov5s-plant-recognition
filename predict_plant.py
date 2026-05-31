# ========== 补充所有必要导入（放在文件最开头） ==========
# ========== 完整导入（含np） ==========
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np  # 新增：导入numpy
from PIL import Image
import os

# ========== 配置 ==========
MODEL_PATH = "models/plant_cls_model.pth"
CLASSES = ["BaJiaoJinPan", "DuJuan", "GuangYuLan", "GuiYe", "HaiTong",
           "MuJin", "ShiNan", "WuTong", "YinXing"]
IMG_SIZE = 224
DEVICE = torch.device("cpu")

# ========== ECA注意力类（依赖np） ==========
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # 此处使用np.log2，必须导入numpy
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

# ========== 模型定义 ==========
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
model = LightWeightClsModel(num_classes=9).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ========== 图像预处理 ==========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== 预测函数 ==========
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"错误：图像路径不存在 → {img_path}")
        return None, 0.0
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        pred_cls = CLASSES[pred_idx]
        pred_prob = torch.softmax(output, dim=1)[0][pred_idx].item() * 100
    print(f"图像：{img_path}")
    print(f"预测类别：{pred_cls}（置信度：{pred_prob:.2f}%）")
    return pred_cls, pred_prob

# ========== 测试预测 ==========
if __name__ == "__main__":
    # 替换为你的实际图像路径
    test_img_path = "data/test/BaJiaoJinPan/BaJiaoJinPan02.jpg"
    predict_image(test_img_path)