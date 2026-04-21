# -*- coding: utf-8 -*-
# SymbolGeneration/Agent/agents/generator_agent.py
import base64
import time
import json
from pathlib import Path
from typing import List, Optional

import requests
from openai import OpenAI
from config import MODELS, OPENAI_API_KEY, IMAGE_SIZE, CREATIVE_SAMPLES
from utils import log, extract_json

client = OpenAI(api_key=OPENAI_API_KEY)

# gpt-image-1-mini 可能对尺寸也有严格限制，我们保守一点
SUPPORTED_SIZES = {"1024x1024", "512x512", "256x256"}

# ==============================================================================
# [Prompt Engineer] LLM-based Prompt Synthesis
# ==============================================================================

SYSTEM_PROMPT_ENGINEER = """
You are an expert Prompt Engineer.
Goal: Translate structured design requirements into a DALL-E 3 prompt.
CRITICAL: 
1. If 'Visual Instruction' is provided (e.g., "reclining"), it OVERRIDES any default assumptions.
2. Style: Flat vector map icon, white background, no shading.
3. Output: Raw prompt text only.
"""


def _synthesize_prompt_with_llm(entity_name: str, style_json: dict, variation_note: str = "") -> str:
    """
    [Reasoning] 使用 LLM 动态融合 VLM 的视觉指令。
    """
    # 1. 提取 VLM 指令 (这是画对姿态的关键)
    visual_instruction = style_json.get("visual_prompt_instruction", "No specific instruction.")

    # 2. 提取 Designer 的构图参数
    composition = style_json.get("composition", {})
    palette = style_json.get("palette", [])

    user_content = f"""
    Target: {entity_name}
    Visual Instruction (VLM Truth): {visual_instruction}
    Viewpoint: {composition.get('viewpoint', 'front')}
    Palette: {', '.join(palette[:3])}

    Task: Write a prompt for a simple flat vector icon.
    """

    try:
        resp = client.chat.completions.create(
            model=MODELS["LLM_MODEL"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ENGINEER},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()[:1000]
    except Exception:
        # 降级：如果 LLM 挂了，用简单的拼接
        return f"A flat vector map icon of {entity_name} depicting {visual_instruction}, white background."


# ==============================================================================
# [Main Logic] Minimalist API Call (针对 gpt-image-1-mini 优化)
# ==============================================================================

def _download_with_retry(url: str, out_path: Path, tries: int = 3, timeout: int = 20) -> bool:
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return True
        except Exception:
            pass
    return False


def run_generator(outline_path: Optional[str],
                  style_json: str,
                  user_text: str = "",
                  structure_spec=None,
                  base_image: Optional[str] = None,
                  mask_image: Optional[str] = None
                  ) -> List[str]:
    """
    生成器 Agent：极简模式，修复 Unknown parameter 错误。
    """
    # 1. 尺寸检查
    target_size = IMAGE_SIZE if IMAGE_SIZE in SUPPORTED_SIZES else "1024x1024"
    n_samples = max(1, int(CREATIVE_SAMPLES))
    target_model = MODELS["IMAGE_MODEL"]

    OUT_DIR = (Path(__file__).resolve().parents[1] / "outputs")
    IMG_DIR = OUT_DIR / "images"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    saved: List[str] = []

    try:
        style_data = extract_json(style_json) or {}
    except:
        style_data = {}

    for i in range(n_samples):
        # 2. 合成 Prompt (关键步骤：融合 VLM 信息)
        print(f"🤖 [Gen-Agent] 正在合成 Prompt (Sample {i + 1})...")
        prompt = _synthesize_prompt_with_llm(user_text, style_data, f"Var {i}")

        (OUT_DIR / f"IconGenerator_prompt_{ts}_{i + 1}.txt").write_text(prompt, encoding="utf-8")
        print(f"📝 [Synthesized Prompt] {prompt[:80]}...")

        out_path = IMG_DIR / f"candidate_{ts}_{i + 1}.png"
        resp = None

        # 3. [T2I] 核心生成逻辑 (极简版)
        if resp is None:
            try:
                # 构造最基础的参数，剔除一切可能导致报错的高级参数
                gen_params = {
                    "model": target_model,
                    "prompt": prompt,
                    "size": target_size,
                    "n": 1
                    # 移除了 response_format，使用 API 默认行为 (通常是 URL)
                    # 移除了 quality, style
                }

                print(f"🎨 调用绘图模型: {target_model} (基础参数)")
                resp = client.images.generate(**gen_params)

            except Exception as e:
                print(f"❌ 生成失败 (Model: {target_model}): {e}")
                continue

        # 4. 保存结果 (处理默认的 URL 返回)
        datum = getattr(resp, "data", [None])[0]
        if not datum: continue

        b64 = getattr(datum, "b64_json", None)
        url = getattr(datum, "url", None)

        # 优先处理 URL，因为我们去掉了 response_format: b64_json
        if url:
            print(f"📥 收到图片URL，正在下载...")
            if _download_with_retry(url, out_path):
                saved.append(str(out_path))
                print(f"🖼️ 已下载本地: {out_path}")
            else:
                print("⚠️ URL 下载失败")
        elif b64:
            out_path.write_bytes(base64.b64decode(b64))
            saved.append(str(out_path))
            print(f"🖼️ 已保存本地(B64): {out_path}")
        else:
            print("⚠️ 无图像数据返回")

        time.sleep(1)  # 给 API 稍微喘口气的机会

    if not saved:
        print("❌ Image API returned no usable images.")
        return []

    log("IconGenerator", f"{len(saved)} images generated.")
    return saved