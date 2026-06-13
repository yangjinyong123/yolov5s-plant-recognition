yolov5s-plant-recognition 9类植物叶片识别项目 README.md

一、项目概述
本项目基于 YOLOv5s 构建9类校园植物叶片智能识别系统，融合多套实验方案：
1. 监督检测识别：轻量化YOLOv5s主干搭配ECA注意力模块，完成9种植物叶片目标检测分类；测试集分类准确率91.11%，CPU环境可高效推理运行。
2. 无监督聚类分析：提取叶片深度特征，PCA降维后使用K-Means完成无监督聚类，聚类匹配准确率87.78%，用于样本分布、特征可分性实验分析。
3. 可视化交互系统：基于PyQt5搭建GUI图形界面，支持单张/批量图片上传、一键训练、实时预测、结果保存导出。
数据集共9类标注叶片样本，整体工程完整覆盖 数据集处理→模型训练→验证评估→批量推理→聚类实验→桌面UI部署 全流程。


二、环境依赖
1. 一键安装依赖
bash
pip install -r requirements.txt

   核心依赖清单
- torch、torchvision（深度学习框架）
- ultralytics、YOLOv5运行依赖库
- opencv-python、matplotlib、seaborn（图像处理/绘图）
- scikit-learn、numpy、pandas（聚类、特征计算、数值处理）
- PyQt5（图形界面GUI）
- pillow、pyyaml、tqdm、scipy等工具库

  运行环境要求
- Python ≥3.9
- Windows/Linux/macOS 全平台兼容
- 推荐NVIDIA GPU加速训练；无GPU可CPU低速训练推理

 
 三、完整项目文件结构
yolov5s-plant-recognition/

├─ 数据集目录
│  ├─ data/                 # 数据集配置缓存
│  ├─ plant_9classes/       # 9类植物原始标注图片+labels标签
│  ├─ test/                  # 测试图片集
│  ├─ val/                   # 验证图片集
│  └─ data.zip               # 压缩打包数据集
├─ 核心训练推理脚本
│  ├─ train.py               # YOLOv5主训练入口（本项目核心）
│  ├─ train_plant_cls.py     # 植物分类专用训练脚本
│  ├─ val.py                 # 验证集mAP、精确率、召回率评估
│  ├─ predict.py             # 基础单张图片预测
│  ├─ predict_plant.py       # 定制化叶片单张识别推理
│  ├─ predict_plant_batch.py # 批量文件夹图片预测
│  └─ predict_plant_visual.py # 预测结果带框可视化保存
├─ 特征聚类实验模块
│  ├─ extract_90_features.py      # 提取90维叶片深度特征向量
│  ├─ split_features_by_class.py  # 按9个类别拆分特征数据
│  ├─ plant_kmeans_clustering.py  # K-Means无监督聚类算法
│  ├─ plant_kmeans_clustering.png # 聚类散点结果图
│  └─ plant_cluster_90.png        # 90维特征聚类可视化图
├─ 修复&工具脚本
│  ├─ fix_label_ids.py       # 标签id修复，解决标注类别越界报错
│  ├─ export.py              # 模型导出（onnx/tensorrt等格式）
│  ├─ hubconf.py             # Torch Hub快速加载模型配置
│  └─ test_pure.py           # 轻量化快速模型加载测试
├─ 界面与配置文件
│  ├─ main_ui.py             # PyQt5可视化交互GUI主程序
│  ├─ plant_data.yaml        # 9类数据集核心配置（类别名、路径、nc数量）
│  ├─ yolov8s-cls.pt         # 预训练权重文件
│  ├─ requirements.txt       # 环境依赖清单
│  ├─ pyproject.toml        # Python项目打包配置
│  ├─ tutorial.ipynb        # Jupyter分步调试教程笔记
│  └─ README.md / README.zh-CN.md # 中英文项目说明
└─ 运行输出文件夹（自动生成）
   ├─ classify/    # 分类推理输出
   └─ models/      # 训练保存best.pt/last.pt权重

四、关键核心文件详解
1. `train.py`（项目最核心，YOLOv5标准训练脚本）
你提供的完整版训练脚本是整个检测模型的训练基座，完整逻辑：
1. 参数解析：设置预训练权重、数据集yaml、迭代轮数、批次、图像尺寸、优化器、早停策略、冻结层数等超参；
2. 环境与设备初始化：自动识别GPU/CPU、支持DDP多卡并行训练；
3.模型加载：加载yolov5s预训练权重，替换适配9类植物的检测头；支持断点续训`--resume`；
4. 数据集加载：读取`plant_data.yaml`路径，构建训练/验证DataLoader，支持马赛克、多尺度、HSV等数据增强；
5. 训练循环
    - Warmup热身学习率、梯度累积、混合精度AMP加速；
    - 损失计算（box框损失+obj置信度损失+cls分类损失）；
    - EMA指数平均权重、梯度裁剪防止爆炸；
6. 每轮验证评估：调用`val.py`计算mAP、Precision、Recall；自动保存最优权重`best.pt`与最新权重`shturl.`；
7. 支持遗传超参进化evolve、早停EarlyStopping、模型优化器剥离保存等高级功能。

基础启动训练命令
bash
基础训练
python train.py --weights yolov5s.pt --data plant_data.yaml --epochs 100 --batch-size 16 --imgsz 640

断点续训
python train.py --resume runs/train/exp/shturl.

CPU强制运行
python train.py --weights yolov5s.pt --data plant_data.yaml --device cpu


 2. `plant_data.yaml`（训练运行必备配置）
定义9类植物名称、训练/验证图片路径、类别总数`nc=9`，train.py、val.py、所有预测脚本全部依赖此文件，路径错误直接运行失败。
示例内部结构：
yaml
nc: 9
names:
  0: 八角金盘
  1: 杜鹃
  2: 银杏
  ... # 剩余6类植物名称
train: ./plant_9classes/train
val: ./plant_9classes/val


3. 推理预测三大核心脚本
1. `predict_plant.py`：单张叶片图片识别，输出类别、置信度；
2. `predict_plant_batch.py`：一次性识别整个文件夹所有叶片图片，批量导出结果；
3. `predict_plant_visual.py`：识别后自动保存绘制检测框的效果图，方便论文插图。

 4. `main_ui.py`（答辩展示加分核心）
PyQt5图形操作界面，无需敲命令行：
- 按钮选择图片/文件夹；
- 一键调用训练、单张预测、批量预测；
- 窗口内展示识别效果图、置信度文字结果；
- 一键导出识别日志与图片，毕业设计演示首选入口。

 5. 聚类实验核心（论文创新点部分）
1. `extract_90_features.py`：从YOLO模型中间层提取每张叶片90维深度特征；
2. `plant_kmeans_clustering.py`：PCA降维+K-Means无监督聚类，生成散点对比图；
3. 输出指标：聚类准确率87.78%，用于论文“无监督特征可分性分析”章节。

6. 辅助关键脚本
- `val.py`：独立验证脚本，单独跑权重测mAP、精确率召回率，对比不同模型优劣；
- `fix_label_ids.py`：标注标签修复工具，标签序号和yaml不匹配时报错专用修复；
- `requirements.txt`：环境部署唯一依赖清单，克隆仓库后第一时间执行安装。


五、分步完整运行教程
 步骤1：克隆仓库&安装环境
 bash
git clone https://github.com/yangjinyong123/yolov5s-plant-recognition.git
cd yolov5s-plant-recognition
pip install -r requirements.txt

步骤2：解压数据集
解压`data.zip`，确认`plant_9classes`文件夹图片+label标签完整，核对`plant_data.yaml`里文件路径和本地路径一致。

步骤3：启动模型训练
bash
python train.py --weights yolov5s.pt --data plant_data.yaml --epochs 100

训练完成后权重保存在 `runs/train/exp/` 下：`best.pt`（最优精度）、`shturl.`（最后一轮）

步骤4：验证模型精度
bash
python val.py --weights runs/train/exp/best.pt --data plant_data.yaml
控制台输出mAP、精确率、召回率、整体准确率91.11%

步骤5：叶片识别推理
1. 单张识别
```bash
python predict_plant.py --weights runs/train/exp/best.pt --img 640 --source test/1.jpg
```
2. 批量文件夹识别
```bash
python predict_plant_batch.py --weights runs/train/exp/best.pt --source test/
```

步骤6：运行GUI可视化界面（演示推荐）
bash
python main_ui.py
弹窗图形界面，鼠标点击即可完成全部操作。

步骤7：运行K-Means聚类特征实验
bash
 1. 提取90维特征
python extract_90_features.py
2. 执行聚类并绘图
python plant_kmeans_clustering.py
运行完毕生成两张聚类PNG图表。


 六、模型性能指标
1. 监督YOLOv5s-ECA检测分类
   - 类别总数：9类校园植物叶片
   - 测试集整体识别准确率：91.11%
   - 支持CPU低配置设备实时推理，轻量化部署友好
2. PCA+K-Means无监督聚类
   - 特征维度：90维深度特征
   - 聚类匹配准确率：87.78%
   - 可视化散点图直观展示9类样本特征分布差异
3. 优化点：原生YOLOv5s嵌入ECA注意力机制，小幅提升小叶片、遮挡叶片识别精度


七、自定义修改指南
1. 更换自己的植物数据集
    - 替换`plant_9classes`图片与labels；
    - 修改`plant_data.yaml`里`nc`类别数量、`names`植物名称、train/val路径；
    - 运行`fix_label_ids.py`校验标签id无越界；
2. 调整训练超参
    在`train.py`启动命令追加参数：
    ```bash
    # 改批次、迭代轮数、图像大小
    python train.py --batch-size 8 --epochs 150 --imgsz 480
    ```
3. 关闭GPU只用CPU
    启动命令加 `--device cpu`
4. 修改GUI界面文字、按钮逻辑
    直接编辑`main_ui.py`内Qt控件代码
5. 调整聚类特征维度、聚类簇数
    修改`extract_90_features.py`与`plant_kmeans_clustering.py`内维度、n_cluster参数



八、开源说明
本项目为期末设计开源学习工程，代码完全开放，允许学习、修改、二次拓展；禁止打包倒卖源码与数据集。
遇到运行报错、路径问题、调参疑问，可提交仓库Issues交流调试。

可选补充（上传GitHub优化）
1. 大文件`data.zip`、`yolov8s-cls.pt`如果上传失败，可在README备注：预训练权重可官网下载，数据集联系获取；
2. 可附带截图：GUI界面效果图、聚类图、训练loss曲线图、识别检测框样例图；



