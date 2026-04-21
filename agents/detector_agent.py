# -*- coding: utf-8 -*-
# agents/detector_agent.py
from __future__ import annotations
import base64, mimetypes, time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
from openai import OpenAI

from config import MODELS, OPENAI_API_KEY
# [修改点 1] 增加导入 extract_json 用于解析模型返回的 JSON
from utils import log, extract_json

client = OpenAI(api_key=OPENAI_API_KEY)

# [修改点 2] 这是一个全新的、强化的 System Prompt
# 目的：强迫视觉模型忽略“情感/意义”，专注于“几何/姿态/构图”
SYSTEM_PROMPT = """
You are a Computer Vision Structure Analyst for cartographic symbol generation.
Your goal is NOT to describe the "meaning" or "emotion" of the image, but to extract its **PHYSICAL GEOMETRY** and **VISUAL STRUCTURE** so a designer can recreate its silhouette perfectly.

**CRITICAL INSTRUCTIONS:**
1. **Ignore context/emotion**: Do not say "symbolizes love" or "looks peaceful".
2. **Focus on Shape**: Describe the bounding box shape (e.g., horizontal rectangle, vertical tower, circle).
3. **Analyze Posture**: If it's a statue/figure, explicitly state the pose (e.g., **reclining**, **lying down**, **seated**, **standing upright**). THIS IS VITAL.
4. **Composition**: Is it a single vertical element? Or a wide horizontal group?
5. **Key Elements**: List the distinct visual components (e.g., "woman lying on right side", "child nestled on left").

**Output Schema (JSON only):**
{
  "entity_type": "statue|bridge|building|tower|other",
  "structure": {
    "composition": "horizontal|vertical|square",
    "posture": "reclining|standing|seated|unknown", 
    "shape_features": [
       "specific physical description 1",
       "specific physical description 2"
    ],
    "dominant_lines": "curved|straight|angular"
  },
  "visual_constraints": {
    "must": ["element 1", "element 2"],
    "must_not": ["element 3"]
  }
}
"""


def _to_data_url(image_path: str | Path) -> str:
    """读取本地图像并转为 data URL（自动识别 jpg/png 等 MIME）"""
    p = Path(image_path)
    img_bytes = p.read_bytes()
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# [修改点 3] 修改了返回类型提示，增强了处理逻辑
def run_detector(image_path: str, schema: str = "") -> Dict[str, Any]:
    """
    识别地标对象（把图片作为 data URL 发送给多模态模型）
    并提取关键的几何与姿态特征。
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"[LandmarkDetector] 图像不存在：{p}")

    data_url = _to_data_url(p)

    # [修改点 4] 使用新的 SYSTEM_PROMPT 和 response_format
    resp = client.chat.completions.create(
        model=MODELS["VISION_MODEL"],
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze the structure of this landmark image."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ]
    )
    content = resp.choices[0].message.content
    log("LandmarkDetector", content)

    # [修改点 5] 解析并重组数据
    # 将 "posture" 和 "composition" 这种关键信息提取出来，
    # 强制插入到 shape_features 的最前面，确保 PromptPlanner 能看到。
    try:
        data = extract_json(content)
        if not data:
            return {}

        # 提取深层结构信息
        struct = data.get("structure", {})
        posture = struct.get("posture", "")
        comp = struct.get("composition", "")

        # 获取特征列表
        features = struct.get("shape_features", [])

        # 关键步骤：把姿态和构图变成显式的特征描述
        if posture and posture != "unknown":
            features.insert(0, f"Posture is {posture}")  # 置顶！
        if comp:
            features.insert(1, f"Composition is {comp}")  # 置顶！

        # 构造返回给 Orchestrator 的标准格式
        result_spec = {
            "entity_type": data.get("entity_type", "landmark"),
            "structure": {
                "structural_system": "visual_extracted",
                "shape_features": features,
                # 根据构图自动建议视角
                "view_recommendation": "side" if comp == "horizontal" else "front"
            },
            "constraints": data.get("visual_constraints", {})
        }
        return result_spec

    except Exception as e:
        print(f"⚠️ Detector 解析失败: {e}")
        return {}


def run_extractor(image_path: str) -> str:
    """提取地标轮廓（Canny + 细化），返回保存的轮廓图路径。"""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"[OutlineExtractor] 图像不存在：{p}")

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"[OutlineExtractor] 读取图像失败：{p}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 80, 180)

    # 细化边缘：闭操作去断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"outline_{time.strftime('%Y%m%d-%H%M%S')}.png"

    cv2.imwrite(str(out_path), edges)
    log("OutlineExtractor", f"Saved outline: {out_path}")
    return str(out_path)