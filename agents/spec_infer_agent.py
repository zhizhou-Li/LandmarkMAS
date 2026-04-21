# SymbolGeneration/Agent/agents/spec_infer_agent.py
from __future__ import annotations
from typing import Any, Dict, Optional, List
from openai import OpenAI
import json
from config import OPENAI_API_KEY, MODELS
from utils import save_json, log, extract_json

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_MSG = (
    "You are a Universal Spec Planner for Cartography."
    "Your goal: Consolidate User Intent and Grounder Knowledge into a strict JSON spec."
    "Rules:\n"
    "1. Trust VLM Visual Facts above all else (if Grounder says 'reclining', structure MUST be 'reclining').\n"
    "2. Use short values, avoid prose.\n"
    "3. Output ONLY JSON."
    "\nSchema:\n"
    "{"
    ' "entity": {"name": str, "location": str},'
    ' "structure": {'
    '    "structural_system": "truss|arch|suspension|reclining_figure|seated_figure|unknown",'
    '    "shape_features": [str],'
    '    "view_recommendation": "front|side|isometric"'
    ' },'
    ' "constraints":{"must":[str], "must_not":[str]},'
    ' "visual_fact_check": "string (summary of VLM findings)"'
    "}"
)


def infer_structure_spec(user_text: str, detector_spec: Optional[str | Dict[str, Any]] = None) -> Dict[str, Any]:
    """保留旧接口，防止其他代码报错"""
    return infer_spec(user_text, {"detector_context": detector_spec})


def infer_spec(user_text: str, grounder_data: dict) -> dict:
    """
    [适配接口] 供 run_multiagent.py 调用。
    负责将 Grounder 搜到的 VLM 事实融合进 Spec。
    """
    print(f"🧠 [SpecInfer] 正在整合知识与视觉事实...")

    # 1. 提取 VLM 视觉事实 (这是闭环修正的关键)
    vlm_info = "No specific visual facts."
    if "vlm_analysis" in grounder_data:
        v = grounder_data["vlm_analysis"]
        vlm_info = f"[VLM EVIDENCE - HIGH PRIORITY]: Posture={v.get('posture')}, Shape={v.get('shape_description')}"

    # 2. 准备上下文
    grounder_summary = json.dumps(grounder_data.get('entity', {}), ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"""
        User Intent: {user_text}
        Grounder Knowledge: {grounder_summary}
        {vlm_info}

        Task: Generate the JSON Spec. Ensure structural_system matches the VLM Evidence.
        """}
    ]

    try:
        resp = client.chat.completions.create(
            model=MODELS["LLM_MODEL"],
            response_format={"type": "json_object"},
            temperature=0.0,
            messages=messages,
        )
        raw = resp.choices[0].message.content
        spec = extract_json(raw) or {}

        # 继承图片引用，方便下游 Reviewer 使用
        if "reference_images" in grounder_data:
            spec["reference_images"] = grounder_data["reference_images"]

        save_json("SpecInfer", spec)
        return spec

    except Exception as e:
        print(f"⚠️ SpecInfer Error: {e}")
        return {"entity": {"name": user_text}, "structure": {}, "constraints": {}}