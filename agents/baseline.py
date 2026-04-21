# SymbolGeneration/Agent/baseline.py
from __future__ import annotations
from pathlib import Path
from SymbolGeneration.Agent.agents.generator_agent import run_generator

BASELINE_DIR = Path(__file__).resolve().parent / "outputs" / "baseline"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

def run_baseline(user_text: str):
    """
    单步基线：
    - 不用结构约束，不用多轮优化
    - 直接根据 user_text 生成 CREATIVE_SAMPLES 张，取第一张作为基线结果
    """
    # run_generator 会自己去用 prompt_planner 生成提示词
    paths = run_generator(
        outline_path=None,
        style_json="{}",           # 给一个空样式 JSON；保持简单
        user_text=user_text,
        structure_spec=None,
        base_image=None,
        mask_image=None
    )
    if not paths:
        raise RuntimeError("Baseline generation failed")
    # 拷贝/重命名到 baseline 专用目录，方便后续统计
    src = Path(paths[0])
    dst = BASELINE_DIR / src.name
    if src != dst:
        dst.write_bytes(src.read_bytes())
    return str(dst)
