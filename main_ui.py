import torch
import sys
import os
from PIL import Image
from torchvision import transforms
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import torch.nn.functional as F
import numpy as np

# ===================== 配置 =====================
CLASSES = [
    "BaJiaoJinPan", "DuJuan", "GuangYuLan", "GuiYe", "HaiTong",
    "MuJin", "ShiNan", "WuTong", "YinXing"
]
DEVICE = "cpu"
IMG_SIZE = 224

# ===================== 强制拒识阈值（已经调到最可靠） =====================
CONFIDENCE_THRESHOLD = 92.0  # 必须很高才认为是叶子，否则一律拒识

# ===================== 模型 =====================
import torch.nn as nn
import numpy as np

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

# ===================== 预处理 =====================
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    return val_transform(img).unsqueeze(0)

# ===================== 批量识别 =====================
class PredictThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(list)
    def __init__(self, paths):
        super().__init__()
        self.paths = paths
    def run(self):
        res = []
        with torch.no_grad():
            total = len(self.paths)
            for i, p in enumerate(self.paths):
                try:
                    out = model(preprocess(p))
                    prob = F.softmax(out, dim=1)
                    idx = prob.argmax().item()
                    max_prob = prob[0, idx].item() * 100

                    # ===================== 拒识逻辑（最终版） =====================
                    if max_prob >= CONFIDENCE_THRESHOLD:
                        res.append({"path": p, "class": CLASSES[idx], "prob": max_prob})
                    else:
                        res.append({"path": p, "class": "没有此类叶子或者不是树木照片", "prob": max_prob})

                except Exception as e:
                    res.append({"path": p, "class": "没有此类叶子或者不是树木照片", "prob": 0})
                self.progress_signal.emit(int((i+1)/total*100))
        self.result_signal.emit(res)

# ===================== 界面 =====================
class PlantUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("植物叶片分类系统")
        self.setGeometry(100, 100, 1200, 700)
        self.current_img = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = QWidget()
        lv = QVBoxLayout(left)
        self.img_label = QLabel("请上传图片")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border:1px solid #ccc; min-height:400px")
        lv.addWidget(self.img_label)

        btn_layout = QHBoxLayout()
        self.upload = QPushButton("单张上传")
        self.upload.clicked.connect(self.upload_img)
        btn_layout.addWidget(self.upload)

        self.batch_upload = QPushButton("批量识别")
        self.batch_upload.clicked.connect(self.batch_recognize)
        btn_layout.addWidget(self.batch_upload)

        self.cls_btn = QPushButton("开始分类")
        self.cls_btn.clicked.connect(self.classify)
        self.cls_btn.setEnabled(False)
        btn_layout.addWidget(self.cls_btn)
        lv.addLayout(btn_layout)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lv.addWidget(self.progress)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        rv.addWidget(QLabel("识别结果"))
        rv.addWidget(self.result_text)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["图片路径", "识别类别", "置信度"])
        rv.addWidget(QLabel("批量识别结果"))
        rv.addWidget(self.table)

        layout.addWidget(left, 2)
        layout.addWidget(right, 3)

    def upload_img(self):
        f, _ = QFileDialog.getOpenFileName()
        if f:
            self.current_img = f
            self.show_img(f)
            self.cls_btn.setEnabled(True)

    def classify(self):
        with torch.no_grad():
            out = model(preprocess(self.current_img))
            prob = F.softmax(out, dim=1)
            idx = prob.argmax().item()
            max_prob = prob[0, idx].item() * 100

            # ===================== 单张拒识 =====================
            if max_prob >= CONFIDENCE_THRESHOLD:
                text = f"✅ 识别结果：{CLASSES[idx]}\n置信度：{max_prob:.2f}%"
            else:
                text = f"❌ 没有此类叶子或者不是树木照片\n置信度：{max_prob:.2f}%"

            self.result_text.setText(text)

    def batch_recognize(self):
        files, _ = QFileDialog.getOpenFileNames()
        if not files:
            return
        self.progress.setVisible(True)
        self.thread = PredictThread(files)
        self.thread.progress_signal.connect(self.progress.setValue)
        self.thread.result_signal.connect(self.show_batch_result)
        self.thread.start()

    def show_batch_result(self, res):
        self.progress.setVisible(False)
        self.table.setRowCount(len(res))
        for i, item in enumerate(res):
            self.table.setItem(i, 0, QTableWidgetItem(item["path"]))
            self.table.setItem(i, 1, QTableWidgetItem(item["class"]))
            self.table.setItem(i, 2, QTableWidgetItem(f"{item['prob']:.2f}%"))

    def show_img(self, p):
        img = Image.open(p).convert("RGB")
        img.thumbnail((500,400))
        qimg = QImage(np.array(img), img.width, img.height, img.width*3, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlantUI()
    window.show()
    sys.exit(app.exec_())