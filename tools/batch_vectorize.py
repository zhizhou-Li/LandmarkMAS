# -*- coding: utf-8 -*-
import os
from pathlib import Path
import sys

# --- 1. 导入你的核心矢量化函数 ---
# 确保脚本能找到 agents 文件夹里的模块
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    # 引用你刚才提供的 pipeline 函数
    from agents.semantic_vectorizer import semantic_vectorization_pipeline

    print("✅ 成功导入矢量化核心模块。")
except ImportError as e:
    print(f"❌ 导入失败，请确保 agents/semantic_vectorizer.py 路径正确: {e}")
    sys.exit(1)

# --- 2. 配置路径 ---
# 源图片目录
SOURCE_DIR = r'Z:\python_projects\map_entropy\SymbolGeneration\Agent\outputs\images\baita\lundun'
# 目标矢量图目录 (在当前路径下创建 vec_3)
TARGET_DIR = os.path.join(SOURCE_DIR, '1')


def batch_process():
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"📂 已创建目标文件夹: {TARGET_DIR}")

    # 支持的图片格式
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    # 获取目录下所有图片文件
    image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(valid_extensions)]

    total = len(image_files)
    print(f"🚀 开始批量转换，共计 {total} 张图片...")
    print("-" * 40)

    success_count = 0
    for index, filename in enumerate(image_files, 1):
        input_path = os.path.join(SOURCE_DIR, filename)

        # 构造输出文件名：原名.svg，保存在 vec_3 目录下
        svg_name = Path(filename).stem + ".svg"
        output_path = os.path.join(TARGET_DIR, svg_name)

        try:
            # 调用你代码中的核心管道
            # simplify_factor=1.5 是你代码中推荐的经验值
            semantic_vectorization_pipeline(
                str(input_path),
                str(output_path),
                simplify_factor=1.5
            )
            success_count += 1
            print(f"[{index}/{total}] ✅ 已完成: {filename} -> {svg_name}")
        except Exception as e:
            print(f"[{index}/{total}] ❌ 失败: {filename}, 错误原因: {e}")

    print("-" * 40)
    print(f"✨ 任务完成！")
    print(f"总计: {total} | 成功: {success_count} | 失败: {total - success_count}")
    print(f"结果保存在: {TARGET_DIR}")


if __name__ == "__main__":
    batch_process()