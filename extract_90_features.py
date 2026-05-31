# -*- coding: utf-8 -*-
"""
最终修复版：适配YOLOv5 v7.0，90张9类植物图片特征提取
解决：fp16_imgsz参数不兼容 + 钩子函数未捕获特征问题
"""
import torch
import numpy as np
import os
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

# YOLOv5模块导入
try:
    from models.common import DetectMultiBackend, SPPF
    from utils.dataloaders import LoadImages
    from utils.general import check_img_size, check_file
    from utils.torch_utils import select_device
except ImportError as e:
    raise ImportError(f"导入YOLOv5模块失败：{e}\n请确保在YOLOv5根目录运行此脚本")

# 全局字典存储特征（避免作用域问题）
feature_storage = {
    'extracted': None,
    'hook_registered': False
}


def feature_hook(module, input_tensor, output_tensor):
    """稳定的特征钩子函数"""
    feature_storage['extracted'] = output_tensor
    feature_storage['hook_registered'] = True


def batch_extract_features(
        weights: str = 'runs/train/exp2/weights/best.pt',
        source: str = 'C:/Users/1/yolov5/data/plant_9classes',
        imgsz: tuple = (640, 640),
        device: str = 'cpu',
        save_path: str = 'batch_plant_features.npy'
) -> dict:
    print("=" * 60)
    print("【YOLOv5 v7.0适配版】90张9类植物图片特征提取")
    print("=" * 60)

    # -------------------------- 1. 基础检查 --------------------------
    # 检查模型文件
    weights = check_file(weights)
    if not os.path.exists(weights):
        print(f"❌ 模型文件不存在：{weights}")
        return None

    # 检查图片文件夹
    if not os.path.exists(source):
        print(f"❌ 图片文件夹不存在：{source}")
        return None

    # 设备初始化
    device = select_device(device)
    print(f"✅ 运行设备：{device}")

    # -------------------------- 2. 加载模型（适配v7.0） --------------------------
    print(f"🔄 加载模型：{weights}")
    # 移除v7.0不支持的fp16_imgsz参数
    model = DetectMultiBackend(
        weights,
        device=device,
        dnn=False,
        fp16=False,
        data=None  # 不加载数据配置
    )
    model.eval()  # 固定评估模式
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    print(f"✅ 模型加载完成 | 步长：{stride} | 输入尺寸：{imgsz}")

    # -------------------------- 3. 递归加载所有图片 --------------------------
    print(f"🔄 加载图片：{source}（递归读取子文件夹）")
    img_formats = {'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'}

    def get_all_image_paths(folder):
        img_paths = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.split('.')[-1].lower() in img_formats:
                    img_paths.append(os.path.join(root, file))
        return img_paths

    all_img_paths = get_all_image_paths(source)
    if not all_img_paths:
        print(f"❌ 未找到任何图片！")
        return None

    total_imgs = len(all_img_paths)
    print(f"✅ 找到有效图片数量：{total_imgs} 张")
    if total_imgs != 90:
        print(f"⚠️ 警告：图片数量不是90张，当前为 {total_imgs} 张")

    # 写入临时文件
    temp_file = 'temp_img_paths.txt'
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_img_paths))

    # 加载数据集（适配v7.0，auto=False）
    dataset = LoadImages(
        temp_file,
        img_size=imgsz,
        stride=stride,
        auto=False,  # 禁用自动tensor转换
        transforms=None
    )

    # -------------------------- 4. 特征层定位 + 钩子注册 --------------------------
    target_layer = None
    hook_handle = None

    # 遍历模型所有层，优先找SPPF（兼容v7.0）
    for name, module in model.model.named_modules():
        if isinstance(module, SPPF) or 'SPPF' in name or 'sppf' in name:
            target_layer = module
            target_layer_name = name
            # 注册钩子
            hook_handle = target_layer.register_forward_hook(feature_hook)
            print(f"✅ 找到SPPF层：{name}，已注册特征钩子")
            break

    # 兜底：找不到SPPF则用模型倒数第5层
    if target_layer is None:
        print("⚠️ 未找到SPPF层，使用模型倒数第5层提取特征")
        layers = list(model.model.named_modules())
        if len(layers) >= 5:
            target_layer_name, target_layer = layers[-5]
            hook_handle = target_layer.register_forward_hook(feature_hook)
            print(f"✅ 注册钩子到层：{target_layer_name}")
        else:
            print("❌ 无法找到有效特征层")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return None

    # -------------------------- 5. 批量提取特征 --------------------------
    all_features = {}
    success_count = 0
    fail_list = []

    pbar = tqdm(
        dataset,
        total=total_imgs,
        desc="提取特征",
        ncols=100,
        colour='green'
    )

    for path, im, im0s, vid_cap, s in pbar:
        img_name = Path(path).name
        pbar.set_postfix({"当前图片": img_name[:20]})

        try:
            # 重置特征存储
            feature_storage['extracted'] = None

            # 图片预处理（手动转换）
            im = torch.from_numpy(im).to(device)
            im = im.float() / 255.0  # 归一化
            if len(im.shape) == 3:
                im = im.unsqueeze(0)  # [1,3,640,640]

            # 前向传播（禁用梯度）
            with torch.no_grad():
                _ = model(im)

            # 检查是否捕获到特征
            if feature_storage['extracted'] is not None:
                # 转换为numpy数组
                feat_np = feature_storage['extracted'].cpu().detach().numpy()

                # 存储特征
                all_features[img_name] = {
                    'image_path': path,
                    'raw_feature': feat_np,
                    'flat_feature': feat_np.reshape(1, -1),
                    'shape': feat_np.shape,
                    'class_folder': Path(path).parent.name
                }
                success_count += 1
            else:
                raise ValueError("钩子函数未捕获到特征")

        except Exception as e:
            fail_list.append(img_name)
            tqdm.write(f"\n❌ 提取失败 {img_name}：{str(e)[:60]}")

    # -------------------------- 6. 清理资源 --------------------------
    # 移除钩子
    if hook_handle is not None:
        hook_handle.remove()
    pbar.close()
    torch.cuda.empty_cache()

    # 清理临时文件
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"🗑️ 清理临时文件：{temp_file}")

    # -------------------------- 7. 保存特征 --------------------------
    if all_features:
        np.save(save_path, all_features)
        file_size = os.path.getsize(save_path) / 1024 / 1024

        print("\n" + "=" * 60)
        print("特征提取完成！汇总信息：")
        print("=" * 60)
        print(f"✅ 成功提取：{success_count} 张")
        print(f"❌ 提取失败：{len(fail_list)} 张")
        if fail_list and len(fail_list) <= 10:
            print(f"   失败列表：{fail_list}")
        elif fail_list:
            print(f"   失败列表：{fail_list[:10]}...（共{len(fail_list)}个）")
        print(f"💾 特征文件：{save_path}")
        print(f"📦 文件大小：{file_size:.2f} MB")
    else:
        print("\n❌ 未提取到任何特征！")
        return None

    return all_features


if __name__ == "__main__":
    config = {
        "weights": "runs/train/exp2/weights/best.pt",
        "source": "C:/Users/1/yolov5/data/plant_9classes",
        "device": "cpu",
        "save_path": "batch_plant_features.npy"
    }

    # 执行提取
    batch_extract_features(**config)