# -*- coding: utf-8 -*-
import os
import subprocess
from PIL import Image

# =================配置区域=================
# 您的输入图片路径
INPUT_DIR = r"/Agent/outputs/images/baita"
# 输出路径
OUTPUT_DIR = os.path.join(INPUT_DIR, "base_vec")
# potrace.exe 的路径 (假设就在当前脚本旁边)
POTRACE_PATH = r"../potrace.exe"


# =========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def raster_to_svg_cmd(input_path, output_path):
    """
    直接指挥 potrace.exe 干活，不需要安装任何 Python 库
    """
    # 中间临时文件
    temp_bmp = input_path.replace(".png", ".bmp").replace(".jpg", ".bmp")

    try:
        # 1. 用 PIL 把图片转成 potrace 能看懂的 BMP 格式
        image = Image.open(input_path)

        # 处理透明背景 -> 变白底
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert("RGB")

        # 二值化 (黑白化)
        gray = image.convert("L")
        threshold = 128
        # 像素值>128变白(0)，<=128变黑(1)
        binary = gray.point(lambda p: 255 if p > threshold else 0)
        binary = binary.convert("1")
        binary.save(temp_bmp)

        # 2. 命令行调用 potrace.exe
        # 相当于在 CMD 里敲命令
        cmd = [POTRACE_PATH, temp_bmp, "-s", "-o", output_path]

        # 隐藏弹出的黑框框 (Windows专属)
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        subprocess.run(cmd, check=True, startupinfo=startupinfo, stderr=subprocess.PIPE)
        return True

    except FileNotFoundError:
        print("❌ 错误: 找不到 potrace.exe！请确认它和脚本在同一个文件夹里。")
        return False
    except Exception as e:
        print(f"❌ 失败 {os.path.basename(input_path)}: {e}")
        return False
    finally:
        # 删掉临时生成的 BMP
        if os.path.exists(temp_bmp):
            try:
                os.remove(temp_bmp)
            except:
                pass


def batch_process():
    ensure_dir(OUTPUT_DIR)

    # 检查 potrace.exe 是否到位
    if not os.path.exists(POTRACE_PATH):
        print(f"❌ 严重错误: 没找到 {POTRACE_PATH}")
        print("请下载 potrace.exe 并把它放到这里：")
        print(os.getcwd())
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total = len(files)

    print(f"🚀 开始转换 (使用官方 Potrace 引擎)，共 {total} 张...")

    count = 0
    for filename in files:
        in_path = os.path.join(INPUT_DIR, filename)
        name_no_ext = os.path.splitext(filename)[0]
        out_path = os.path.join(OUTPUT_DIR, name_no_ext + ".svg")

        print(f"   正在处理: {filename} ...", end="\r")
        if raster_to_svg_cmd(in_path, out_path):
            count += 1

    print(f"\n✅ 全部完成! 成功: {count}/{total}")
    print(f"📂 矢量图在: {OUTPUT_DIR}")


if __name__ == "__main__":
    batch_process()