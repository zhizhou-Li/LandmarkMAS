import cv2
import numpy as np
import svgwrite
from skimage import measure
from shapely.geometry import LineString, Polygon
from shapely.validation import make_valid


def load_and_preprocess(image_path):
    """
    预处理：精准提取黑色线条区域
    """
    # 【核心修改】使用 numpy 读取二进制数据，以支持中文路径
    try:
        # np.fromfile 能够正确处理 Windows 下的中文路径
        img_array = np.fromfile(image_path, dtype=np.uint8)
        # 使用 cv2.imdecode 从内存中解码图片
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        raise FileNotFoundError(f"读取图片失败 (可能包含特殊字符): {image_path}, 错误: {e}")

    if img is None:
        raise FileNotFoundError(f"无法解码图片（文件可能损坏或路径错误）: {image_path}")
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

    # 简单去噪
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # 【重要修改】增加 1px Padding，确保图像边缘的线条能够闭合
    # 如果不加这个，底座贴底的部分会变成非闭合线条，填充时会出错
    binary_mask_padded = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)

    return binary_mask_padded > 0


def extract_contours_as_shapely(binary_mask, tolerance=1.0):
    """
    提取轮廓并简化
    """
    contours = measure.find_contours(binary_mask, 0.5)
    valid_geoms = []

    for contour in contours:
        # 坐标翻转 (row, col) -> (x, y)
        coords = np.fliplr(contour)

        # 【重要修改】因为我们在预处理时加了 1px 的 padding
        # 为了还原原始位置，这里需要把坐标减去 1
        coords -= 1

        if len(coords) < 3: continue

        try:
            # 创建 LineString 进行简化
            line = LineString(coords)
            if tolerance > 0:
                line = line.simplify(tolerance, preserve_topology=True)

            if not line.is_empty:
                valid_geoms.append(line)
        except:
            continue

    return valid_geoms


def semantic_vectorization_pipeline(image_path, output_svg_path, simplify_factor=1.0, mode='polygon'):
    """
    全要素矢量化管道
    """
    # 1. 预处理
    binary_mask = load_and_preprocess(image_path)
    h, w = binary_mask.shape  # 注意这是 padding 后的尺寸

    # 2. 几何提取
    # 注意：这里我们传入的是 padding 后的 mask，但在内部提取时已经减去了偏移量
    geoms = extract_contours_as_shapely(binary_mask, tolerance=simplify_factor)

    # 3. 导出 SVG
    # 使用原始图片尺寸（padding 后的尺寸减去 2）
    dwg = svgwrite.Drawing(output_svg_path, profile='tiny', size=(w - 2, h - 2))

    # 【核心样式修改】
    # 改为黑色填充，无描边，且使用 evenodd 规则处理镂空（如眼睛）
    path_style = {
        'fill': 'black',
        'stroke': 'none',
        'fill_rule': 'evenodd'
    }

    # 绘制白色背景
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))

    # 构建 Path 数据字符串
    path_data_list = []

    for geom in geoms:
        coords = list(geom.coords)
        if len(coords) < 2: continue

        # 构建 SVG Path 指令: M (移动) -> L (画线) -> ... -> Z (闭合)
        # 即使原始图形是不闭合的线，Z 命令也会强制连接首尾，形成封闭区域以便填充
        d = [f"M {coords[0][0]:.2f} {coords[0][1]:.2f}"]
        for x, y in coords[1:]:
            d.append(f"L {x:.2f} {y:.2f}")
        d.append("Z")  # 强制闭合

        path_data_list.append(" ".join(d))

    # 将所有轮廓合并到一个 <path> 标签中
    # 这样 evenodd 规则才能在“大轮廓”包“小轮廓”时生效
    full_path_d = " ".join(path_data_list)

    if full_path_d:
        dwg.add(dwg.path(d=full_path_d, **path_style))

    dwg.save()
    print(f"Saved optimized SVG to: {output_svg_path}")

    return output_svg_path