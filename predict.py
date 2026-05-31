# predict.py
import os
import argparse
import torch
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image

# 9类植物配置
CLASSES = [
    "BaJiaoJinPan", "DuJuan", "GuangYuLan", "GuiYe", "HaiTong",
    "MuJin", "ShiNan", "WuTong", "YinXing"
]
NUM_CLASSES = len(CLASSES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 默认模型路径
DEFAULT_MODEL_PATH = "models/yolov5s_eca_9cls.pt"
# 图像预处理配置
IMG_SIZE = 224


# 加载模型
def load_model(model_path):
    """加载训练好的9类植物分类模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    model = YOLO(model_path)
    model.to(DEVICE)
    model.eval()
    print(f"模型加载完成，使用设备：{DEVICE}")
    return model


# 单张图像预测
def predict_single_image(model, img_path, save_result=False, save_dir="predict_results"):
    """
    单张图像预测
    :param model: 加载好的YOLO模型
    :param img_path: 图像路径
    :param save_result: 是否保存预测结果
    :param save_dir: 结果保存目录
    :return: 预测结果字典
    """
    # 校验图像路径
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图像文件不存在：{img_path}")

    # 预测
    results = model(img_path, imgsz=IMG_SIZE)

    # 解析结果
    pred_cls_idx = results[0].probs.top1  # 最高概率类别索引
    pred_cls = CLASSES[pred_cls_idx]  # 最高概率类别名称
    pred_prob = results[0].probs.top1conf.item() * 100  # 最高概率值
    # 所有类别概率
    all_probs = {CLASSES[idx]: round(prob.item() * 100, 2) for idx, prob in enumerate(results[0].probs.data)}

    # 结果字典
    result_dict = {
        "image_path": img_path,
        "predicted_class": pred_cls,
        "confidence(%)": pred_prob,
        "all_classes_prob": all_probs
    }

    # 打印结果
    print("\n===== 单张图像预测结果 =====")
    print(f"图像路径：{img_path}")
    print(f"预测类别：{pred_cls}（置信度：{pred_prob:.2f}%）")
    print("9类植物置信度分布：")
    for cls, prob in all_probs.items():
        print(f"  {cls}: {prob}%")

    # 保存结果
    if save_result:
        os.makedirs(save_dir, exist_ok=True)
        # 保存结果到CSV
        result_df = pd.DataFrame([result_dict])
        csv_path = os.path.join(save_dir, "single_predict_result.csv")
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        # 保存带预测结果的图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 添加文字标注
        cv2.putText(img, f"Pred: {pred_cls} ({pred_prob:.2f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # 保存图像
        img_save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(img_save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"\n预测结果已保存至：{save_dir}")

    return result_dict


# 批量图像预测
def predict_batch_images(model, img_dir, save_result=True, save_dir="predict_results"):
    """
    批量图像预测
    :param model: 加载好的YOLO模型
    :param img_dir: 图像目录（包含待预测的所有图像）
    :param save_result: 是否保存预测结果
    :param save_dir: 结果保存目录
    :return: 批量预测结果列表
    """
    # 校验目录
    if not os.path.isdir(img_dir):
        raise NotADirectoryError(f"图像目录不存在：{img_dir}")

    # 获取所有图像文件
    img_ext = [".jpg", ".jpeg", ".png", ".bmp"]
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                 if os.path.splitext(f)[1].lower() in img_ext]
    if len(img_paths) == 0:
        raise ValueError(f"目录{img_dir}下未找到图像文件（支持格式：{img_ext}）")

    # 批量预测
    batch_results = []
    print(f"\n开始批量预测，共{len(img_paths)}张图像...")
    for img_path in tqdm(img_paths, desc="批量预测进度"):
        results = model(img_path, imgsz=IMG_SIZE)
        pred_cls_idx = results[0].probs.top1
        pred_cls = CLASSES[pred_cls_idx]
        pred_prob = results[0].probs.top1conf.item() * 100
        all_probs = {CLASSES[idx]: round(prob.item() * 100, 2) for idx, prob in enumerate(results[0].probs.data)}

        batch_results.append({
            "image_path": img_path,
            "predicted_class": pred_cls,
            "confidence(%)": round(pred_prob, 2),
            "all_classes_prob": all_probs
        })

    # 打印汇总结果
    print("\n===== 批量预测汇总结果 =====")
    cls_count = {cls: 0 for cls in CLASSES}
    for res in batch_results:
        cls_count[res["predicted_class"]] += 1
    print("各类别预测数量：")
    for cls, count in cls_count.items():
        print(f"  {cls}: {count}张")

    # 保存结果
    if save_result:
        os.makedirs(save_dir, exist_ok=True)
        # 保存详细结果到CSV
        # 转换all_classes_prob为单独列
        result_df = pd.DataFrame(batch_results)
        prob_df = pd.DataFrame(result_df["all_classes_prob"].tolist())
        result_df = pd.concat([result_df.drop("all_classes_prob", axis=1), prob_df], axis=1)
        csv_path = os.path.join(save_dir, "batch_predict_result.csv")
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n批量预测结果已保存至：{csv_path}")

    return batch_results


# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="9类植物叶片分类预测脚本")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="模型文件路径（默认：models/yolov5s_eca_9cls.pt）")
    parser.add_argument("--img_path", type=str, default=None,
                        help="单张图像路径（单张预测时使用）")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="图像目录（批量预测时使用）")
    parser.add_argument("--save_result", action="store_true", default=True,
                        help="是否保存预测结果（默认：保存）")
    parser.add_argument("--save_dir", type=str, default="predict_results",
                        help="结果保存目录（默认：predict_results）")
    return parser.parse_args()


# 主函数
if __name__ == "__main__":
    # 解析参数
    args = parse_args()

    # 加载模型
    model = load_model(args.model_path)

    # 执行预测
    if args.img_path:
        # 单张预测
        predict_single_image(model, args.img_path, args.save_result, args.save_dir)
    elif args.img_dir:
        # 批量预测
        predict_batch_images(model, args.img_dir, args.save_result, args.save_dir)
    else:
        # 无参数时，默认单张预测示例（需替换为实际图像路径）
        print("未指定图像路径/目录，执行示例预测...")
        sample_img_path = "data/test/YinXing/1.jpg"  # 替换为实际测试图像路径
        if os.path.exists(sample_img_path):
            predict_single_image(model, sample_img_path, save_result=True)
        else:
            print(f"示例图像不存在：{sample_img_path}，请指定--img_path或--img_dir参数")