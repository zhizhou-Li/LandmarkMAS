# -*- coding: utf-8 -*-
# Z:\python_projects\map_entropy\SymbolGeneration\Agent\SASR.py
import os
import sys
import shutil
from pathlib import Path

# 确保项目根目录在路径中
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from Agent.agents.grounder_agent import ground_entity_to_spec
from Agent.agents.designer_agent import run_designer, refine_designer
from Agent.agents.reviewer_agent import run_reviewer
from Agent.agents.generator_agent import run_generator  # 原始生成器
from Agent.utils import log, save_json, extract_json, OUTPUT_DIR  # 这里的 OUTPUT_DIR 是 3.23


def smart_generator_wrapper(*args, **kwargs):
    """
    包装器：在不修改原代码前提下，将生成的文件从默认的 outputs 移动到 3.23 目录
    """
    # 1. 调用原始生成器 (它会把文件存到 Agent/outputs)
    saved_paths = run_generator(*args, **kwargs)
    if not saved_paths:
        return []

    # 2. 准备 3.23 目标目录
    target_img_dir = OUTPUT_DIR / "images"
    target_img_dir.mkdir(parents=True, exist_ok=True)

    new_paths = []
    # 3. 移动图像文件 (.png)
    for p in saved_paths:
        old_p = Path(p)
        new_p = target_img_dir / old_p.name
        if old_p.exists():
            shutil.move(str(old_p), str(new_p))  # 移动文件
            new_paths.append(str(new_p))

    # 4. 移动提示词记录文件 (.txt)
    # 原始生成器会将 .txt 放在 Agent/outputs 目录下
    old_out_dir = Path(saved_paths[0]).parents[1]
    for txt_file in old_out_dir.glob("*.txt"):
        shutil.move(str(txt_file), str(OUTPUT_DIR / txt_file.name))

    return new_paths


def run_ablation_experiment(landmark_name, iterations=3):
    print(f"🚀 启动消融对比实验: {landmark_name}")
    print(f"📁 结果将通过 Wrapper 自动重定向至: {OUTPUT_DIR}")

    # 1. 统一检索
    structure_spec = ground_entity_to_spec(landmark_name)
    ref_url = structure_spec.get("reference_image_url")

    # --- 路径 A: LS-MAS ---
    print("\n--- 正在运行 [LS-MAS] 路径 ---")
    current_style = run_designer(landmark_name, "{}", structure_spec)
    ls_mas_final_img = None
    for i in range(iterations):
        # 使用包装后的生成器
        imgs = smart_generator_wrapper(None, current_style, landmark_name, structure_spec)
        if not imgs: break
        ls_mas_final_img = imgs[0]

        review_result = run_reviewer(ls_mas_final_img, ref_url, landmark_name)
        if review_result.get("decision") == "PASS": break
        current_style = refine_designer(current_style, review_result, structure_spec)

    # --- 路径 B: SASR ---
    print("\n--- 正在运行 [SASR] 路径 ---")
    sasr_style = run_designer(landmark_name, "{}", structure_spec)
    sasr_final_img = None
    for i in range(iterations):
        # 使用包装后的生成器
        imgs = smart_generator_wrapper(None, sasr_style, landmark_name, structure_spec)
        if not imgs: break
        sasr_final_img = imgs[0]

        mock_self_review = {"decision": "FAIL", "critique": "Self-refinement...", "scores": {"accuracy": 7}}
        sasr_style = refine_designer(sasr_style, mock_self_review, structure_spec)

    print(f"\n✅ 实验完成！所有 LS-MAS 和 SASR 结果均已保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    run_ablation_experiment("The Kelpies sculptures 苏格兰凯尔派马头,只用简单的线条，黑白二值化。")