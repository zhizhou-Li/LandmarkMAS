# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os


def generate_baseline_svg(image_path, output_svg_path):
    """
    模拟 Baseline 矢量化：直接将栅格边缘转化为 SVG，不进行任何简化。
    这种方法会产生大量的冗余节点（锯齿），正好用于展示 Ours 方法的优化效果。
    """
    # 1. 读取图像
    # 处理中文路径：使用 imdecode 代替 imread
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    except Exception:
        img = None

    if img is None:
        print(f"❌ 无法读取图片 (可能路径有误): {os.path.basename(image_path)}")
        return None

    # 2. 预处理 (转灰度 -> 二值化)
    # 如果有透明通道，先转为白色背景
    if len(img.shape) == 3 and img.shape[2] == 4:
        # 分离 alpha 通道
        b, g, r, a = cv2.split(img)
        # 创建白底
        background = 255 * np.ones_like(b, dtype=np.uint8)
        # 融合
        alpha = a.astype(float) / 255
        b = cv2.multiply(1.0 - alpha, background.astype(float)) + cv2.multiply(alpha, b.astype(float))
        g = cv2.multiply(1.0 - alpha, background.astype(float)) + cv2.multiply(alpha, g.astype(float))
        r = cv2.multiply(1.0 - alpha, background.astype(float)) + cv2.multiply(alpha, r.astype(float))
        img = cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 使用 OTSU 阈值
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 提取原始轮廓 (Raw Contours)
    # CHAIN_APPROX_NONE 意味着保留所有轮廓点，不压缩 -> 产生海量节点
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    total_nodes = 0
    svg_paths = []

    # 4. 手动构建 SVG 路径字符串
    height, width = binary.shape

    for cnt in contours:
        # cnt 是一个 (N, 1, 2) 的数组
        points = cnt.reshape(-1, 2)
        if len(points) < 3: continue  # 忽略极小噪点

        total_nodes += len(points)

        # 构建 'M x y L x y L x y ... Z' 指令
        # 注意：这里生成的点是极其密集的
        path_data = "M " + " L ".join([f"{p[0]},{p[1]}" for p in points]) + " Z"
        svg_paths.append(f'<path d="{path_data}" fill="none" stroke="red" stroke-width="1" />')

    # 5. 写入 SVG 文件
    svg_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <g id="layer1">
        {''.join(svg_paths)}
    </g>
</svg>
"""

    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    return total_nodes


def batch_process_baseline(input_dir):
    # 1. 设置输出目录 (在输入目录下建一个子文件夹)
    output_dir = os.path.join(input_dir, "baseline_vec_raw")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 已创建输出目录: {output_dir}")

    # 2. 获取所有图片文件
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

    print(f"🚀 开始批量生成 Baseline SVG (未优化/多节点版)...")
    print(f"📂 扫描路径: {input_dir}")
    print(f"📊 共发现 {len(files)} 张图片\n")

    stats = []

    # 3. 循环处理
    for filename in files:
        file_path = os.path.join(input_dir, filename)

        # 构造输出文件名 (原名_baseline.svg)
        name_no_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name_no_ext}_baseline.svg")

        print(f"⚙️ 正在处理: {filename} ...", end="")

        nodes = generate_baseline_svg(file_path, output_path)

        if nodes is not None:
            print(f" ✅ 完成 | 节点数 (V_raw): {nodes}")
            stats.append({"file": filename, "v_raw": nodes})
        else:
            print(f" ❌ 失败")

    # 4. 打印统计摘要 (可以直接用于论文表格)
    print("\n" + "=" * 50)
    print("📝 数据统计摘要 (请复制到 Excel/论文表格)")
    print("=" * 50)
    print(f"{'文件名':<40} | {'V_raw (原始节点数)':<15}")
    print("-" * 60)

    total_v_raw = 0
    for item in stats:
        print(f"{item['file']:<40} | {item['v_raw']:<15}")
        total_v_raw += item['v_raw']

    if len(stats) > 0:
        avg_v_raw = total_v_raw / len(stats)
        print("-" * 60)
        print(f"平均节点数 (Average V_raw): {int(avg_v_raw)}")
    print("=" * 50)
    print(f"📂 所有 SVG 文件已保存在: {output_dir}")


# ================= 使用示例 =================
if __name__ == "__main__":
    # 【配置您的文件夹路径】
    # 注意：路径前面加 r 防止转义，且确保路径存在
    target_dir = r"/Agent/outputs/images/Geo/1.29/round_3"

    if os.path.exists(target_dir):
        batch_process_baseline(target_dir)
    else:
        print(f"❌ 错误: 找不到路径 {target_dir}")