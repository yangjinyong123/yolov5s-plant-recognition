import os

# 配置你的数据集标签目录
label_dirs = [
    r"C:\Users\1\plant_detection\labels\train",
    r"C:\Users\1\plant_detection\labels\val"
]
# 9类的合法ID范围：0-8
max_legal_id = 8

# 遍历所有标注文件
for dir_path in label_dirs:
    print(f"\n=== 处理目录：{dir_path} ===")
    for filename in os.listdir(dir_path):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(dir_path, filename)
        lines = []
        has_error = False

        # 读取文件并修正ID=9的行
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f.readlines(), 1):
                line = line.strip()
                if not line:
                    lines.append(line)
                    continue

                parts = line.split()
                try:
                    cls_id = int(parts[0])
                    if cls_id == 9:
                        # 找到ID=9的行，提示并替换（这里默认替换为8，你可改为0-8中的任意值）
                        print(f"❌ {filename} 第{line_num}行：ID=9 → 自动替换为8")
                        new_line = "8 " + " ".join(parts[1:])  # 替换ID为8，保留坐标
                        lines.append(new_line)
                        has_error = True
                    elif cls_id > max_legal_id:
                        print(f"❌ {filename} 第{line_num}行：ID={cls_id} 超出0-8范围 → 自动替换为8")
                        new_line = "8 " + " ".join(parts[1:])
                        lines.append(new_line)
                        has_error = True
                    else:
                        lines.append(line)
                except ValueError:
                    print(f"❌ {filename} 第{line_num}行：ID不是数字 → {parts[0]}")
                    lines.append(line)
                    has_error = True

        # 保存修正后的文件
        if has_error:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"✅ {filename} 已修正并保存\n")

print("所有文件检查/修正完成！")