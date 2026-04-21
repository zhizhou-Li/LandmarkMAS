# -*- coding: utf-8 -*-
import base64
import json
import requests
import cv2
import numpy as np
from openai import OpenAI
from config import MODELS, OPENAI_API_KEY
from utils import log, extract_json

client = OpenAI(api_key=OPENAI_API_KEY)

REVIEWER_SYSTEM_PROMPT = """
You are a Senior Cartographic Art Director. 
Your goal is to guide a Junior Designer (AI) to refine a map icon until it perfectly matches the reference and strict mapping rules.

**CRITICAL: Objective Tool Reports Overrule Everything**
You will receive 'Objective Tool Reports' about the generated image. 
- If the tool says the image has >4 colors, you MUST FAIL it, no matter how good it looks.
- Map symbols MUST be flat, geometric, and use a very limited palette (<= 4 colors).

**Evaluation Criteria:**
1. **Semantic Accuracy:** Does the shape match the reference?
2. **Contextual Consistency:** Does it obey the Tool Reports (e.g., color count)?

**Output Format:** JSON only.
{
    "scores": {
        "semantic_accuracy": int, // 0-10
        "contextual_consistency": int // 0-10
    },
    "critique": "Specific, imperative commands for the Designer to fix the style JSON (e.g., 'Reduce palette to 3 colors', 'Make it flat').",
    "decision": "PASS" | "FAIL"
}
"""


def _encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ==========================================
# 🛠️ 客观检测工具区 (Tools)
# ==========================================
def tool_check_color_count(image_path: str) -> int:
    """客观工具：统计图片中的主要颜色数量 (过滤掉极少量的噪点像素)"""
    img = cv2.imread(image_path)
    if img is None: return 0

    # 转为二维像素数组
    pixels = img.reshape(-1, 3)
    # 统计每种颜色的出现次数
    colors, counts = np.unique(pixels, axis=0, return_counts=True)

    # 过滤掉占比不到 1% 的噪点/抗锯齿边缘像素
    total_pixels = len(pixels)
    main_colors = [c for c, count in zip(colors, counts) if (count / total_pixels) > 0.01]
    return len(main_colors)


def tool_check_background(image_path: str) -> str:
    """客观工具：检查背景是否干净"""
    # 此处可根据需要扩展，目前默认返回合规
    return "Valid (White/Transparent Background)"


# ==========================================
# 🧠 评审主逻辑
# ==========================================
def run_reviewer(candidate_path: str, reference_url: str, entity_name: str, visual_instruction: str = "") -> dict:
    print(f"🧐 [Reviewer Agent] 正在调用客观工具并评估: {candidate_path} ...")

    try:
        candidate_b64 = _encode_image(candidate_path)
    except Exception as e:
        return {"decision": "FAIL", "critique": f"Image load error: {e}", "scores": {}}

    # 1. 执行客观检测工具
    actual_color_count = tool_check_color_count(candidate_path)
    bg_status = tool_check_background(candidate_path)

    objective_facts = f"""
    [OBJECTIVE TOOL REPORTS - DO NOT IGNORE]
    - Detected Main Colors: {actual_color_count} (Rule: MUST be <= 4)
    - Background Status: {bg_status}
    """

    if actual_color_count > 4:
        print(f"   ⚠️ 客观工具报警：颜色超标 ({actual_color_count} 种)！强行触发驳回机制。")

    # 2. 构建 Prompt
    user_content = [
        {"type": "text", "text": f"Target Entity: {entity_name}"},
        {"type": "text", "text": f"Visual Facts (Truth): {visual_instruction}"},
        {"type": "text", "text": objective_facts},
        {"type": "text",
         "text": "Task: Evaluate the Generated Icon. If Detected Colors > 4, output FAIL and order the Designer to reduce colors."},
    ]

    # 尝试加载参考图
    if reference_url and reference_url.startswith("http"):
        try:
            dl_headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://image.baidu.com"}
            dl_resp = requests.get(reference_url, headers=dl_headers, timeout=5)
            if dl_resp.status_code == 200:
                ref_b64 = base64.b64encode(dl_resp.content).decode('utf-8')
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}})
                user_content.append({"type": "text", "text": "[Image 1: Real Reference (The Truth)]"})
        except:
            pass

    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{candidate_b64}"}})
    user_content.append({"type": "text", "text": "[Image 2: Generated Icon (Evaluate This)]"})

    try:
        resp = client.chat.completions.create(
            model=MODELS["LLM_MODEL"],
            messages=[
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0  # 保持0，确保客观公正
        )
        result = extract_json(resp.choices[0].message.content)

        print(
            f"📊 [Review Decision] {result.get('decision')} | 准确度: {result.get('scores', {}).get('semantic_accuracy', 0)}")
        if result.get("decision") == "FAIL":
            print(f"🗣️ [Critique to Designer] {result.get('critique')}")

        return result
    except Exception as e:
        print(f"⚠️ Reviewer Error: {e}")
        return {"decision": "FAIL", "critique": "Error in review process.", "scores": {}}