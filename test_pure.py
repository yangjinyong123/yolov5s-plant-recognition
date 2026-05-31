import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

# --------------- 完全复刻训练代码 ---------------
CLASSES = [
    "BaJiaoJinPan", "DuJuan", "GuangYuLan", "GuiYe", "HaiTong",
    "MuJin", "ShiNan", "WuTong", "YinXing"
]
DEVICE = "cpu"
IMG_SIZE = 224

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
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

class LightWeightClsModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), ECA(16),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), ECA(32),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), ECA(64),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), ECA(128),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), ECA(256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# 加载模型
model = LightWeightClsModel(9)
model.load_state_dict(torch.load("models/plant_cls_model.pth", map_location=DEVICE))
model.eval()

# 验证集预处理（和训练一致）
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== 在这里替换成你的八角金盘图片完整路径 ==========
img_path = r"C:\Users\1\yolov5\my-project\test.jpg"

with torch.no_grad():
    img = Image.open(img_path).convert("RGB")
    img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)
    out = model(img_tensor)
    pred_idx = torch.argmax(out, dim=1).item()
    print(f"预测类别索引: {pred_idx}")
    print(f"预测类别名称: {CLASSES[pred_idx]}")