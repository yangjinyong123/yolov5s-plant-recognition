import torch
import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ===================== 【1: 完全照搬你的训练代码】 =====================
CLASSES = [
    "BaJiaoJinPan", "DuJuan", "GuangYuLan", "GuiYe", "HaiTong",
    "MuJin", "ShiNan", "WuTong", "YinXing"
]
DEVICE = "cpu"
IMG_SIZE = 224

# ECA注意力
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

# 你的模型：LightWeightClsModel 完全一样！
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

# ===================== 【2: 加载你的权重】 =====================
model = LightWeightClsModel(num_classes=9)
model.load_state_dict(torch.load("models/plant_cls_model.pth", map_location=DEVICE))
model.eval()

# ===================== 【3: 预处理和训练完全一致！】 =====================
from torchvision import transforms

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = val_transform(img).unsqueeze(0)
    return img.to(DEVICE)

# ===================== 识别线程 =====================
class PredictThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(list)
    def __init__(self, paths):
        super().__init__()
        self.paths = paths
    def run(self):
        res = []
        with torch.no_grad():
            for i, p in enumerate(self.paths):
                try:
                    out = model(preprocess(p))
                    prob = F.softmax(out, dim=1)
                    idx = prob.argmax().item()
                    res.append({"path": p, "class": CLASSES[idx], "prob": prob[0, idx].item()*100})
                except:
                    res.append({"path": p, "class": "识别失败", "prob": 0.0})
                self.progress_signal.emit(int((i+1)/len(self.paths)*100))
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

        self.batch = QPushButton("批量上传")
        self.batch.clicked.connect(self.batch_upload)
        btn_layout.addWidget(self.batch)

        self.cls_btn = QPushButton("开始分类")
        self.cls_btn.clicked.connect(self.classify)
        self.cls_btn.setEnabled(False)
        btn_layout.addWidget(self.cls_btn)
        lv.addLayout(btn_layout)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lv.addWidget(self.progress)

        right = QTabWidget()
        result = QWidget()
        rl = QVBoxLayout(result)
        g1 = QGroupBox("识别结果")
        gl = QVBoxLayout(g1)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        gl.addWidget(self.result_text)
        rl.addWidget(g1)

        g2 = QGroupBox("批量结果")
        bl = QVBoxLayout(g2)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["路径", "类别", "置信度"])
        bl.addWidget(self.table)
        rl.addWidget(g2)
        right.addTab(result, "分类结果")
        right.addTab(QWidget(), "模型性能")
        right.addTab(QWidget(), "错误样本")

        layout.addWidget(left, 2)
        layout.addWidget(right, 3)

    def upload_img(self):
        f, _ = QFileDialog.getOpenFileName()
        if f:
            self.current_img = f
            self.show_img(f)
            self.cls_btn.setEnabled(True)

    def batch_upload(self):
        fs, _ = QFileDialog.getOpenFileNames()
        if fs:
            self.progress.setVisible(True)
            self.t = PredictThread(fs)
            self.t.progress_signal.connect(self.progress.setValue)
            self.t.result_signal.connect(self.show_batch)
            self.t.start()

    def classify(self):
        with torch.no_grad():
            out = model(preprocess(self.current_img))
            prob = F.softmax(out, dim=1)
            idx = prob.argmax().item()
            text = f"✅ 识别结果：{CLASSES[idx]}\n置信度：{prob[0,idx].item()*100:.2f}%\n\n"
            for i, c in enumerate(CLASSES):
                text += f"{c}: {prob[0,i].item()*100:.2f}%\n"
            self.result_text.setText(text)

    def show_img(self, p):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        scale = min(500/w, 400/h)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        qimg = QImage(img.data, img.shape[1], img.shape[0], ch*img.shape[1], QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg))

    def show_batch(self, res):
        self.progress.setVisible(False)
        self.table.setRowCount(len(res))
        for i, r in enumerate(res):
            self.table.setItem(i, 0, QTableWidgetItem(r["path"]))
            self.table.setItem(i, 1, QTableWidgetItem(r["class"]))
            self.table.setItem(i, 2, QTableWidgetItem(f"{r['prob']:.2f}"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlantUI()
    window.show()
    sys.exit(app.exec_())