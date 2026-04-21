# 文件：Agent/core/state.py
from typing import TypedDict, Optional, Dict, List, Any


class SymbolGenerationState(TypedDict):
    # 1. 初始输入
    user_input: str

    # 2. 认知与规划层数据
    intent_schema: Optional[Dict[str, Any]]
    entity_name: str
    grounder_spec: Optional[Dict[str, Any]]
    final_spec: Optional[Dict[str, Any]]
    vlm_fact_str: str
    ref_url: str

    # 3. 设计层数据
    current_style_json: Optional[Dict[str, Any]]

    # 4. 迭代生成与评估层数据
    round_idx: int
    candidate_paths: List[str]
    best_candidate_path: Optional[str]
    critique: Optional[str]  # Reviewer 给出的修改意见
    decision: Optional[str]  # "PASS" 或 "FAIL"
    acc_score: int  # 准确性得分

    # 5. 最终输出
    final_svg_path: Optional[str]