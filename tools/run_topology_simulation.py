# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import svgwrite
from skimage import measure
from shapely.geometry import LineString, MultiLineString

# ================= 配置 =================
IMAGE_DIR = r"Z:\python_projects\map_entropy\SymbolGeneration\Agent\outputs\images\生成"
# 输出目录改为 svg_output_paper_method 以示区别，代表这是严格符合论文方法的
SVG_OUT_DIR = os.path.join(IMAGE_DIR, "svg_output_paper_method")

if not os.path.exists(SVG_OUT_DIR):
    os.makedirs(SVG_OUT_DIR)


# ================= 核心算法函数 (对齐 Agent) =================

def load_and_preprocess(img_array):
    """
    预处理：精准提取黑色线条区域
    【核心修改】增加 np.pad 给图片加一圈白边，确保所有线条都能形成闭合回路
    """
    # 解码图片
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    if img is None: return None

    # 1. 针对带 Alpha 通道的图片
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        is_black = (b < 100) & (g < 100) & (r < 100)
        is_opaque = a > 50
        binary_mask = is_black & is_opaque
    # 2. 针对普通 JPG/PNG
    else:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # OTSU 反向阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_mask = binary > 0

    binary_mask = binary_mask.astype(np.uint8)

    # 形态学闭运算（缝合断裂）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 【关键步骤】给 mask 四周加一圈 0 (背景)，强制所有贴边的图形闭合
    # 这样 find_contours 就能绕着最外圈走回来，形成封闭多边形
    binary_mask_padded = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)

    return binary_mask_padded > 0


def extract_contours_to_shapely(binary_mask, tolerance=1.0):
    """
    使用 skimage 和 shapely 提取并简化线条
    """
    # skimage.measure.find_contours 能更好地处理像素级拓扑
    contours = measure.find_contours(binary_mask, 0.5)
    valid_lines = []

    for contour in contours:
        # skimage 返回 (row, col)，需要翻转为 (x, y)
        coords = np.fliplr(contour)

        # 忽略过短的噪点
        if len(coords) < 3: continue

        try:
            line = LineString(coords)
            # 几何简化 (Douglas-Peucker)
            if tolerance > 0:
                line = line.simplify(tolerance, preserve_topology=True)

            if not line.is_empty:
                valid_lines.append(line)
        except:
            continue

    return valid_lines


def save_lines_to_svg(lines, width, height, filename):
    """
    使用 svgwrite 保存矢量图
    【核心修改】从“描边模式”改为“填充模式” (fill-rule="evenodd")
    """
    # 注意：因为预处理加了1px padding，这里画布尺寸最好不用动，或者输出时微调
    # 为了简单，直接用传入的宽高即可，SVG视口会自动适应
    dwg = svgwrite.Drawing(filename, profile='tiny', size=(width, height))

    # 【关键设置】
    # fill='black': 填充黑色
    # stroke='none': 不要描边（去掉那个难看的双线框）
    # fill_rule='evenodd': 奇偶填充规则。
    #   这决定了：最外层轮廓是黑的，里面的洞（眼睛）是白的，洞里的东西又是黑的。
    path_style = {
        'fill': 'black',
        'stroke': 'none',
        'fill_rule': 'evenodd'
    }

    # 可选：白色背景
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))

    # 我们需要把所有的线条合并成一个巨大的 Path 字符串
    # 这样 evenodd 规则才能在所有形状之间生效
    path_data = []

    for geom in lines:
        if geom.is_empty: continue

        # 提取坐标点
        coords = list(geom.coords)
        if len(coords) < 3: continue  # 忽略不成形的点

        # 【坐标修正】因为 load_and_preprocess 加了 1px padding
        # 这里为了对齐原图，可以把坐标减去 1 (如果不介意1px偏移，不减也可以)
        coords = [(x - 1, y - 1) for x, y in coords]

        # 构建 Path 指令：M(移动到起点) -> L(画线到...) -> Z(闭合)
        # M x0 y0
        d_str = [f"M {coords[0][0]:.2f} {coords[0][1]:.2f}"]
        # L x1 y1 ...
        for x, y in coords[1:]:
            d_str.append(f"L {x:.2f} {y:.2f}")
        # Z (强制闭合形状)
        d_str.append("Z")

        path_data.append(" ".join(d_str))

    # 将所有轮廓拼接成一个长字符串
    full_path_d = " ".join(path_data)

    # 写入 SVG
    if full_path_d:
        dwg.add(dwg.path(d=full_path_d, **path_style))

    dwg.save()


# ================= 主流程 =================

def process_and_export_aligned():
    print(f"📂 Reading images from: {IMAGE_DIR}")
    print(f"💾 Saving PAPER-ALIGNED SVGs to: {SVG_OUT_DIR}")

    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg'))]
    count = 0

    for f in files:
        img_path = os.path.join(IMAGE_DIR, f)

        # 1. 读取 (支持中文路径)
        try:
            img_array = np.fromfile(img_path, dtype=np.uint8)
            # 注意：这里只读取 array，解码在 load_and_preprocess 内部做
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

        # 2. 预处理 (OTSU + Morphology)
        try:
            binary_mask = load_and_preprocess(img_array)
            if binary_mask is None: continue

            # 获取原图尺寸用于 SVG header
            h, w = binary_mask.shape[:2]
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

        # 3. 提取与简化 (Skimage + Shapely)
        # simplify_factor=1.5 是经验值，对应 Agent 代码中的 tolerance
        lines = extract_contours_to_shapely(binary_mask, tolerance=1.5)

        # 4. 导出 SVG
        svg_filename = os.path.splitext(f)[0] + ".svg"
        save_path = os.path.join(SVG_OUT_DIR, svg_filename)

        save_lines_to_svg(lines, w, h, save_path)

        count += 1
        # 简单进度打印
        if count % 5 == 0:
            print(f"✅ Processed {count} images...")

    print(f"\n🎉 Done! Generated {count} SVG files in {SVG_OUT_DIR}")


if __name__ == "__main__":
    process_and_export_aligned()