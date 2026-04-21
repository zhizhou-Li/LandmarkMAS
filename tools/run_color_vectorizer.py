# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import svgwrite
from skimage import measure
from shapely.geometry import LineString, Polygon

# ================= 配置区域 =================
INPUT_IMAGE_PATH = r"/Agent/outputs/images/yuantu/1/Gemini_Generated_Image_nwld77nwld77nwld.png"
OUTPUT_SVG_PATH = INPUT_IMAGE_PATH.replace(".png", "_clean_vectorized.svg")

# 色彩数量
NUM_COLORS = 6

# 【新增配置】最小噪点阈值 (像素)
# 任何小于这个面积的孤立色块都会被删掉！
# 建议值: 20 ~ 100 (根据图片分辨率调整，图片越大，这个值越大)
MIN_AREA_THRESHOLD = 50


# ===========================================

def apply_cartoon_effect(img):
    """
    【改进1】使用 Mean Shift 滤波代替双边滤波
    这能更强力地抹平颜色，产生“卡通化”效果，大幅减少边缘杂色
    """
    print("🎨 正在应用 Mean Shift 滤波 (平坦化颜色)...")
    # sp: 空间窗半径, sr: 颜色窗半径
    # 值越大，画面越平整，细节越少
    return cv2.pyrMeanShiftFiltering(img, sp=15, sr=40)


def remove_small_objects(mask, min_size):
    """
    【改进2】面积过滤
    剔除 Mask 中小于 min_size 的孤立噪点
    """
    # 计算连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 创建一个新 mask
    clean_mask = np.zeros_like(mask)

    # 遍历所有连通域 (从1开始，0是背景)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # 只有面积达标的才保留
        if area >= min_size:
            clean_mask[labels == i] = 255
        # 否则这个区域就被丢弃了（变成黑）

    return clean_mask


def quantize_image(image, k=4):
    data = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    label_map = label.reshape((image.shape[:2]))
    return center, label_map


def extract_contours_as_shapely(binary_mask, tolerance=1.0):
    contours = measure.find_contours(binary_mask, 0.5)
    valid_geoms = []
    for contour in contours:
        coords = np.fliplr(contour)
        coords -= 1  # 抵消 padding
        if len(coords) < 3: continue
        try:
            line = LineString(coords)
            if tolerance > 0:
                line = line.simplify(tolerance, preserve_topology=True)
            if not line.is_empty:
                valid_geoms.append(line)
        except:
            continue
    return valid_geoms


def process_clean_vectorization(input_path, output_path, k=5, min_area=50):
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到文件 {input_path}")
        return

    # 1. 读取
    img_bgr = cv2.imread(input_path)
    if img_bgr is None: return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. 【改进点】预处理：卡通化
    img_smooth = apply_cartoon_effect(img_rgb)

    # 3. K-Means 聚类
    print(f"🧩 正在聚类颜色 (K={k})...")
    centers, labels = quantize_image(img_smooth, k=k)

    # 4. 初始化 SVG
    h, w = img_rgb.shape[:2]
    dwg = svgwrite.Drawing(output_path, profile='tiny', size=(w, h))

    count_layers = 0
    print("⚙️ 开始分层矢量化 (含噪点剔除)...")

    for i, color in enumerate(centers):
        r, g, b = color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        # 简单的背景过滤 (可选)
        if r > 250 and g > 250 and b > 250:
            print(f"   - 跳过背景色: {hex_color}")
            continue

        # 生成原始 Mask
        mask = (labels == i).astype(np.uint8) * 255

        # 【改进点】执行面积过滤 (除噪)
        # 这步操作会把那些“小红点”直接抹掉
        mask_clean = remove_small_objects(mask, min_size=min_area)

        # 如果这一层被过滤光了(全是噪点)，就跳过
        if not np.any(mask_clean):
            print(f"   - ⚠️ 图层 {i} 全是噪点，已忽略。")
            continue

        # 缝隙修复 (膨胀) - 仅对干净的 mask 操作
        kernel = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask_clean, kernel, iterations=1)

        # 闭合 Padding
        mask_padded = np.pad(mask_dilated > 0, pad_width=1, mode='constant', constant_values=0)

        # 提取矢量
        geoms = extract_contours_as_shapely(mask_padded, tolerance=1.0)

        if not geoms: continue

        print(f"   - 处理图层 {i}: {hex_color} (保留主要色块)")

        # 写入 SVG
        group = dwg.g(fill=hex_color, stroke='none', fill_rule='evenodd')
        path_data_list = []
        for geom in geoms:
            coords = list(geom.coords)
            if len(coords) < 2: continue
            d = [f"M {coords[0][0]:.2f} {coords[0][1]:.2f}"]
            for x, y in coords[1:]:
                d.append(f"L {x:.2f} {y:.2f}")
            d.append("Z")
            path_data_list.append(" ".join(d))

        full_d = " ".join(path_data_list)
        if full_d:
            group.add(dwg.path(d=full_d))
            dwg.add(group)
            count_layers += 1

    dwg.save()
    print("-" * 30)
    print(f"🎉 成功！已生成干净的矢量图: {output_path}")
    print(f"   已自动剔除小于 {min_area} 像素的噪点。")


if __name__ == "__main__":
    process_clean_vectorization(INPUT_IMAGE_PATH, OUTPUT_SVG_PATH, k=NUM_COLORS, min_area=MIN_AREA_THRESHOLD)