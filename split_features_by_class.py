# -*- coding: utf-8 -*-
"""
按类别拆分90张植物图片的特征文件
前提：已运行extract_90_features.py生成batch_plant_features.npy
"""
import numpy as np
import os
from pathlib import Path


def split_features_by_class(
        total_feat_path='batch_plant_features.npy',  # 90张总特征文件路径
        img_root='C:/Users/1/yolov5/data/plant_9classes',  # 9类图片根文件夹
        save_root='plant_class_features'  # 分类特征保存文件夹
):
    """按类别拆分特征文件"""
    # 1. 创建保存文件夹
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    print(f"✅ 创建分类特征保存文件夹：{save_root}")

    # 2. 检查总特征文件是否存在
    if not os.path.exists(total_feat_path):
        print(f"❌ 总特征文件不存在：{total_feat_path}")
        print("   请先运行extract_90_features.py提取90张图片的特征！")
        return None

    # 3. 加载总特征文件
    try:
        total_features = np.load(total_feat_path, allow_pickle=True).item()
        print(f"✅ 加载总特征文件成功，共 {len(total_features)} 张图片特征")
    except Exception as e:
        print(f"❌ 加载总特征文件失败：{e}")
        return None

    # 4. 获取9类文件夹名称
    if not os.path.exists(img_root):
        print(f"❌ 图片根文件夹不存在：{img_root}")
        return None

    class_folders = []
    for f in os.scandir(img_root):
        if f.is_dir():
            class_folders.append(f.name)

    if not class_folders:
        print(f"❌ 在 {img_root} 下未找到任何类别子文件夹！")
        return None
    print(f"✅ 识别到 {len(class_folders)} 个类别：{class_folders}")

    # 5. 按类别拆分特征
    split_summary = {}
    for class_name in class_folders:
        class_feat = {}
        # 遍历当前类别的所有图片
        class_img_dir = os.path.join(img_root, class_name)
        if not os.path.exists(class_img_dir):
            print(f"⚠️ 类别文件夹 {class_img_dir} 不存在，跳过")
            continue

        # 获取当前类别的所有图片名
        class_imgs = []
        img_formats = {'jpg', 'png', 'jpeg', 'tif', 'bmp'}
        for f in os.scandir(class_img_dir):
            if f.is_file() and f.name.split('.')[-1].lower() in img_formats:
                class_imgs.append(f.name)

        # 匹配特征
        for img_name in class_imgs:
            if img_name in total_features:
                class_feat[img_name] = total_features[img_name]

        # 保存当前类别的特征
        class_feat_path = os.path.join(save_root, f"{class_name}_features.npy")
        np.save(class_feat_path, class_feat)
        split_summary[class_name] = len(class_feat)
        print(f"   ✅ 类别 {class_name}：提取 {len(class_feat)} 张特征 → {class_feat_path}")

    # 6. 生成汇总文件
    summary_path = os.path.join(save_root, "class_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("9类植物特征拆分汇总\n")
        f.write("=" * 30 + "\n")
        total_split = 0
        for cls, count in split_summary.items():
            f.write(f"{cls}: {count} 张\n")
            total_split += count
        f.write("=" * 30 + "\n")
        f.write(f"总计拆分：{total_split} 张\n")

    print(f"\n✅ 类别汇总文件已保存：{summary_path}")
    print("=" * 50)
    print("特征拆分完成！汇总：")
    print("=" * 50)
    for cls, count in split_summary.items():
        print(f"   {cls}: {count} 张")
    print(f"   总计：{sum(split_summary.values())} 张")


if __name__ == "__main__":
    # 配置参数（根据实际路径调整）
    config = {
        "total_feat_path": "batch_plant_features.npy",
        "img_root": "C:/Users/1/yolov5/data/plant_9classes",
        "save_root": "plant_class_features"
    }
    # 执行拆分
    split_features_by_class(**config)