# -*- coding: utf-8 -*-
# 文件位置: Z:\python_projects\map_entropy\SymbolGeneration\Agent\test_vectorizer_standalone.py
import os
import sys
from pathlib import Path

# --- 1. 路径设置 ---
# 确保脚本能找到 agents 文件夹里的模块
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    # 尝试导入修复后的语义矢量化模块
    # 注意：这里假设 semantic_vectorizer.py 位于 agents/ 子目录下
    from agents.semantic_vectorizer import semantic_vectorization_pipeline
    print("✅ 成功导入 agents.semantic_vectorizer 模块。")
except ImportError as e:
    print("\n❌ 导入失败！请检查以下几点：")
    print("1. 确认本脚本放在了 'Agent' 根目录下。")
    print("2. 确认 'agents/semantic_vectorizer.py' 文件存在。")
    print(f"错误信息: {e}")
    sys.exit(1)

# --- 2. 测试函数 ---
def test_single_image(input_path, simplify_val=1.5):
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到输入文件: {input_path}")
        return

    # 定义输出路径（加个 _test 后缀，避免覆盖原文件）
    input_p = Path(input_path)
    output_path = input_p.with_name(input_p.stem + "_test_fixed.svg")

    print("-" * 40)
    print(f"📐 启动独立矢量化测试...")
    print(f"📥 输入图片: {input_path}")
    print(f"⚙️ 简化因子 (Tolerance): {simplify_val}")
    print("-" * 40)

    try:
        # === 核心调用 ===
        # 这里调用的是你刚才修复过的那个管道函数
        semantic_vectorization_pipeline(str(input_path), str(output_path), simplify_factor=simplify_val)

        # 验证结果
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print("\n🎉 测试成功！")
            print(f"✅ 已生成拓扑安全的矢量图: {output_path}")
            print("   请用浏览器或 AI 打开查看效果。")
        else:
            print("\n❌ 测试失败：输出文件未创建或为空。")

    except Exception as e:
        print("\n❌ 测试过程中发生异常：")
        import traceback
        traceback.print_exc()
        print("\n👉 如果看到 TopologyException，说明修复逻辑没生效，请检查 semantic_vectorizer.py 代码。")


if __name__ == "__main__":
    # ==========================================
    # 👇 请在这里填入你刚才报错的那张图片的绝对路径
    # ==========================================
    target_image = r"Z:\python_projects\map_entropy\SymbolGeneration\Agent\outputs\images\candidate_20251019-175414_1.png"

    # 运行测试
    test_single_image(target_image)