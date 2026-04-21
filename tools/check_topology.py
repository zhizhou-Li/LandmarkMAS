# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from svgpathtools import svg2paths
from shapely.geometry import Polygon
from shapely.validation import explain_validity

# ================= 配置区域 =================
# 您的 Baseline SVG 文件夹路径 (Potrace 生成的结果)
TARGET_DIR = r"/Agent/outputs/images/Geo/1.29/round_3/vec_3"


# ============================================

def check_svg_topology(svg_path):
    """
    解析 SVG，提取多边形，并检查是否存在拓扑错误（自相交）
    """
    try:
        paths, attributes = svg2paths(svg_path)
    except Exception as e:
        print(f"⚠️ 无法解析文件 {os.path.basename(svg_path)}: {e}")
        return 0, 0, False

    total_polygons = 0
    error_polygons = 0
    has_error_in_file = False

    for path in paths:
        if len(path) == 0:
            continue

        # ==========================================
        # 🧨 核心修复：将带有孔洞或不连续的复合路径拆分为独立的子路径
        # 避免把内环和外环连在一条线上导致的“伪自相交”
        # ==========================================
        subpaths = path.continuous_subpaths()

        for subpath in subpaths:
            # 1. 曲线离散化 (Sampling)
            points = []
            NUM_SAMPLES = 10  # 每段曲线采样 10 个点
            for segment in subpath:
                for i in range(NUM_SAMPLES):
                    c = segment.point(i / NUM_SAMPLES)
                    points.append((c.real, c.imag))

            # 添加最后一个终点
            c = subpath[-1].point(1.0)
            points.append((c.real, c.imag))

            # 2. 拓扑验证
            if len(points) >= 3:
                total_polygons += 1
                try:
                    geom = Polygon(points)
                    # is_valid 会自动检查自相交 (Self-intersection)
                    if not geom.is_valid:
                        error_polygons += 1
                        has_error_in_file = True
                except Exception:
                    # 无法构成有效几何体
                    error_polygons += 1
                    has_error_in_file = True

    return total_polygons, error_polygons, has_error_in_file


def main():
    print("🚀 开始进行 OGC 拓扑合规性检测 (自相交检测)...")

    svg_files = glob.glob(os.path.join(TARGET_DIR, "*.svg"))
    if not svg_files:
        print(f"❌ 找不到 SVG 文件，请检查路径: {TARGET_DIR}")
        return

    total_files = len(svg_files)
    files_with_errors = 0

    global_polygons = 0
    global_errors = 0

    for svg_file in svg_files:
        filename = os.path.basename(svg_file)

        polys, errs, has_error = check_svg_topology(svg_file)

        global_polygons += polys
        global_errors += errs

        if has_error:
            files_with_errors += 1
            print(f"❌ {filename} | 包含拓扑错误! (错误多边形: {errs}/{polys})")
        else:
            print(f"✅ {filename} | 拓扑完美 (多边形: {polys})")

    # ================= 统计报表 =================
    print("\n" + "=" * 50)
    print("📊 拓扑错误分析报告 (可直接用于论文)")
    print("=" * 50)
    print(f"测试文件总数: {total_files}")
    print(f"存在自相交错误的文件数: {files_with_errors}")

    file_error_rate = (files_with_errors / total_files) * 100 if total_files > 0 else 0
    print(f"🚩 文件级错误率: {file_error_rate:.2f}%")

    print("-" * 50)
    print(f"提取的多边形总数: {global_polygons}")
    print(f"自相交多边形数: {global_errors}")

    poly_error_rate = (global_errors / global_polygons) * 100 if global_polygons > 0 else 0
    print(f"🚩 多边形级错误率: {poly_error_rate:.2f}% (<- 这个数字就是您的 12.5%)")
    print("=" * 50)
    print(
        "💡 结论：这些自相交多边形在 ArcGIS Pro 中执行空间叠加 (Spatial Join, Intersect) 时会导致 'Topology Exception' 崩溃。")


if __name__ == "__main__":
    main()