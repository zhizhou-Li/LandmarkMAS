# -*- coding: utf-8 -*-
# SymbolGeneration/Agent/agents/designer_agent.py
from openai import OpenAI
from config import MODELS, OPENAI_API_KEY
from utils import log, save_json, extract_json
import json

client = OpenAI(api_key=OPENAI_API_KEY)

# [Upgrade] 增强了 Schema，增加了 composition 字段用于控制视点
STYLE_SCHEMA_HINT = """
Output ONLY a JSON object. Include fields like:
{
  "style_name": "string",
  "composition": {
      "viewpoint": "front|side|isometric|top",
      "angle": "flat|45_degree|low_angle",
      "crop": "whole_structure|distinctive_part"
  },
  "stroke": {"width": number, "pattern": "solid|dash", "corner": "round|miter"},
  "fill": {"type": "none|flat", "opacity": 0-1}, // Removed 'gradient' option to enforce flat style
  "palette": ["#RRGGBB", "..."],
  "simplification": {"tolerance_px": number, "max_points": number},
  "iconography": {"emphasis": ["outline|verticality|truss_structure|polygonal_top_chord"], "negative_space": true/false},
  "visual_prompt_instruction": "string" // A detailed instruction for the image generator. MUST include viewpoint, key structural elements, and strict style constraints (e.g. 'Draw a flat 2D icon...')."
}
"""

# --- 新增：需要强行剔除的颜色词列表 ---
#FORBIDDEN_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "pink", "gold", "golden"]


def _sanitize_style_json(style_str: str, structure_spec) -> str:
    try:
        sj = extract_json(style_str) or {}
        # ... (之前的代码保持不变) ...

        # 3. 填充透明度归一化 (保持不变)
        fill = sj.get("fill") or {}
        # ... (保持不变) ...
        if fill.get("type") == "gradient":
            fill["type"] = "flat"
        sj["fill"] = fill
        instruction = sj.get("visual_prompt_instruction", "")

        # 1. 自动清洗可能引发幻觉或破坏“纯平制图风格”的非规范词汇
        anti_prompts = ["3d", "realistic", "shadow", "gradient", "photorealistic", "lighting"]
        for word in anti_prompts:
            # 简单去除这些破坏制图规则的词
            instruction = instruction.replace(word, "")
            instruction = instruction.replace(word.capitalize(), "")

        # 2. 强化论文中强调的“扁平矢量”底线
        if "flat vector" not in instruction.lower():
            sj["visual_prompt_instruction"] = instruction.strip() + ", flat 2d vector map symbol, limited iconic palette, solid fill only."
        else:
            sj["visual_prompt_instruction"] = instruction

        # 3. 强制截断调色板，严格对齐论文中 "use a very limited palette (<= 4 colors)" 的主张
        if "palette" in sj and isinstance(sj["palette"], list):
            if len(sj["palette"]) > 4:
                print(f"⚠️ [Designer Agent] 调色板颜色超标 ({len(sj['palette'])}种)，系统强制截断至前4种以符合制图规范。")
                sj["palette"] = sj["palette"][:4]

        return json.dumps(sj, ensure_ascii=False)
    except Exception:
        return style_str




def run_designer(landmark_json: str, schema: str, structure_spec=None) -> str:
    # 1. 解析结构约束
    spec_text = json.dumps(structure_spec, ensure_ascii=False) if structure_spec else "{}"

    # 2. 提取多视角视觉参考
    visual_refs_desc = "No specific visual references provided."
    if structure_spec and isinstance(structure_spec, dict):
        refs = structure_spec.get("reference_images", {})
        if refs:
            visual_refs_desc = "Available Visual References (use these to decide viewpoint):\n"
            if refs.get("front"): visual_refs_desc += f"- Front View: {refs['front']}\n"
            if refs.get("side"): visual_refs_desc += f"- Side View: {refs['side']}\n"
            if refs.get("isometric"): visual_refs_desc += f"- Isometric/3D View: {refs['isometric']}\n"

    # 3. 构建 System Prompt (这里融合了“智能视点规则”和“风格强约束”)
    system_prompt = (
        "You are a Cartographic Symbol Designer. "
        "Your goal is to design a minimal, recognizable map icon based on structural constraints and visual references.\n\n"

        "**SMART VIEWPOINT RULES (CRITICAL):**\n"
        "- **Horizontal Rings:** If the object is a flat ring on the ground/water (e.g., Laguna Garzon Bridge), you **MUST** use 'top-down' view to emphasize the map footprint and suppress vertical legs/piers.\n"
        "- **Vertical Rings:** If the object is a standing ring (e.g., Guangzhou Circle, Ferris Wheel), you **MUST** use 'front' view to show the opening clearly.\n"
        "- **Complex Arches:** If the object has crossing, diagonal, or asymmetric arches (e.g., JK Bridge), use 'isometric' view to reveal the 3D spatial relationship.\n"
        "- **Default:** For general buildings, prefer 'front' (elevation) or 'isometric' based on feature visibility.\n\n"
'''
        "**STYLE RULES (NON-NEGOTIABLE):**\n"
        "- **NO** gradients, shadows, or 3D lighting effects.\n"
        "- **NO** photorealistic textures.\n"
        "- **MUST** be flat, 2D vector style.\n"
        "- **MUST** use high-contrast outlines (black/dark grey)."
'''
        "**🎨 COLOR & STYLE RULES:**\n"
        "- **Use Iconic Colors:** You SHOULD use the landmark's real-world colors (e.g., 'International Orange' for Golden Gate, 'White' for Taj Mahal, 'Red' for Temple).\n"
        "- **Flat Style:** Use 'flat vector style' with solid fills. **NO gradients**, NO realistic lighting, NO shading.\n"
        "- **Limited Palette:** Restrict the design to 3-5 distinct colors to maintain clarity.\n"
        "- **Outlines:** You may use dark outlines (black/grey) or colored outlines depending on the style."
    )

    # 4. 构建 General Rule (保留这个！防止马头变怪兽)
    general_rule = (
        "**GENERAL RULE:** When describing sculptures or landmarks, strictly describe the **material and geometry** "
        "(e.g., 'steel plates', 'rivets', 'geometric blocks'). Avoid using biological or organic metaphors "
        "(e.g., 'scales', 'skin', 'monster', 'rider') unless the landmark is a literal statue of a human. "
        "Focus on the architectural construction."
    )

    # 5. 组合 User Prompt
    user_prompt = (
        f"Landmark Info:\n{landmark_json}\n\n"
        f"Interpreter Schema:\n{schema}\n\n"
        f"Structural Constraints (STRICT):\n{spec_text}\n\n"
        f"{visual_refs_desc}\n\n"  # 注入视觉参考
        f"{general_rule}\n\n"  # 注入去生物化规则 (这里必须保留!)
        f"Task: Generate a JSON style sheet.\n"
        f"{STYLE_SCHEMA_HINT}"
    )

    resp = client.chat.completions.create(
        model=MODELS["LLM_MODEL"],
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    content = resp.choices[0].message.content
    log("SymbolDesigner_raw", content)

    # 清洗和验证
    content = _sanitize_style_json(content, structure_spec)
    save_json("SymbolDesigner_json", extract_json(content) or {})
    return content


def refine_designer(prev_style_json: str, review_data: dict, structure_spec=None) -> str:
    spec_text = json.dumps(structure_spec, ensure_ascii=False) if structure_spec else "{}"

    visual_refs_desc = ""
    if structure_spec and isinstance(structure_spec, dict):
        refs = structure_spec.get("reference_images", {})
        if refs:
            visual_refs_desc = f"Visual References: {json.dumps(refs)}"

    # [修正] 这里的 System Prompt 必须与 run_designer 保持一致的严格程度
    system_prompt = (
        "You are an expert Cartographic Designer. Your task is to **UPDATE** the previous JSON style sheet based on the Reviewer's critique.\n"
        "**CRITICAL INSTRUCTIONS:**\n"
        "1. **Preserve Success:** Keep all style parameters and visual descriptions that were NOT criticized.\n"
        "2. **Fix Failures:** Rewrite *only* the parts mentioned in the critique.\n"
        "3. **Cumulative Optimization:** The new 'visual_prompt_instruction' must combine the original correct details with the new corrections.\n"
        "4. **Output Format:** Output ONLY valid JSON.\n\n"

        "**SMART VIEWPOINT RULES (CRITICAL - DO NOT CHANGE THESE):**\n"
        "- **Horizontal Rings:** If the object is a flat ring (e.g., Laguna Garzon), **KEEP** 'top-down' view to prevent it from looking like a trampoline. Suppress vertical legs.\n"
        "- **Vertical Rings:** If the object is a standing ring (e.g., Guangzhou Circle), **KEEP** 'front' view.\n"
        "- **Complex Arches:** If the object has crossing/diagonal arches (e.g., JK Bridge), **KEEP** 'isometric' view to show the 3D structure."
    )

    resp = client.chat.completions.create(
        model=MODELS["LLM_MODEL"],
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Previous style JSON:\n{prev_style_json}"},
            {"role": "assistant", "content": f"Reviewer Feedback:\n{json.dumps(review_data)}\n\n"
                                             f"Structural Constraints:\n{spec_text}\n"
                                             f"{visual_refs_desc}"},
            {"role": "user", "content": STYLE_SCHEMA_HINT}
        ]
    )
    content = resp.choices[0].message.content
    log("SymbolDesigner_refined_raw", content)
    content = _sanitize_style_json(content, structure_spec)
    save_json("SymbolDesigner_refined_json", extract_json(content) or {})
    return content