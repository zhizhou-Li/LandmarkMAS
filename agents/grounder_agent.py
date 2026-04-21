# -*- coding: utf-8 -*-
# 文件路径: SymbolGeneration/Agent/agents/grounder_agent.py
from __future__ import annotations
import json, re, requests, io
from typing import Dict, Any, Optional, List, Tuple
from bs4 import BeautifulSoup
from openai import OpenAI
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch
import base64

from utils import log, save_json, extract_json
from config import OPENAI_API_KEY, MODELS

# --- Global Models ---
# 加载 CLIP 模型
print("⏳ Loading CLIP Model (clip-ViT-B-32)...")
try:
    CLIP_MODEL = SentenceTransformer('clip-ViT-B-32')
    print("✅ CLIP Model Loaded.")
except Exception as e:
    print(f"❌ Failed to load CLIP Model: {e}")
    CLIP_MODEL = None

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Endpoints ---
WIKI_SEARCH = "https://{lang}.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"


# ==============================================================================
# [Core Engine] Multi-View Visual Reasoning & Semantic Anchoring (底层工具库保持不变)
# ==============================================================================

def _search_baidu_candidates(keyword: str, limit: int = 30) -> List[str]:
    print(f"🔎 [Baidu Raw] 正在挖掘 '{keyword}' 的视觉素材库 (Limit: {limit})...")
    url = "https://image.baidu.com/search/acjson"
    params = {
        "tn": "resultjson_com", "logid": "8305096434442765369", "ipn": "rj", "ct": "201326592",
        "is": "", "fp": "result", "queryWord": keyword, "cl": "2", "lm": "-1", "ie": "utf-8",
        "oe": "utf-8", "adpicid": "", "st": "-1", "z": "", "ic": "0", "hd": "", "latest": "",
        "copyright": "", "word": keyword, "s": "", "se": "", "tab": "", "width": "", "height": "",
        "face": "0", "istype": "2", "qc": "", "nc": "1", "fr": "", "expermode": "", "force": "",
        "pn": "0", "rn": str(limit), "gsm": "1e",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/plain, */*; q=0.01", "Referer": "https://image.baidu.com/search/index",
        "X-Requested-With": "XMLHttpRequest",
    }
    candidates = []
    try:
        proxies = {
            "http": None,
            "https": None,
        }
        res = requests.get(url, params=params, headers=headers, proxies=proxies, timeout=6)
        if res.status_code == 200:
            try:
                json_str = res.text.replace(r"\'", "'")
                data = json.loads(json_str)
                if "data" in data and isinstance(data["data"], list):
                    for item in data["data"]:
                        if not isinstance(item, dict): continue
                        img_url = item.get("thumbURL") or item.get("middleURL")
                        if img_url:
                            candidates.append(img_url)
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ 搜图网络错误: {e}")
    return list(set(candidates))


def _multi_view_clip_selection(query_text: str, candidate_urls: List[str]) -> Dict[str, str]:
    if not candidate_urls or CLIP_MODEL is None:
        return {}

    print(f"🧠 [CLIP-3D] 正在构建 {query_text} 的多视角空间认知模型...")
    valid_images = []
    valid_urls = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in candidate_urls:
        try:
            r = requests.get(url, headers=headers, timeout=2)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content))
                if img.mode != 'RGB': img = img.convert('RGB')
                w, h = img.size
                if w > 150 and h > 150:
                    valid_images.append(img)
                    valid_urls.append(url)
        except:
            continue
        if len(valid_images) >= 25: break

    if not valid_images: return {}

    view_prompts = {
        "front": f"A direct front view photo of {query_text}, symmetrical facade, elevation",
        "side": f"A side profile view of {query_text}, lateral perspective",
        "isometric": f"An isometric 45-degree view of {query_text}, 3D structure",
        "top_down": f"A direct top-down aerial map view of {query_text}, satellite image, plan view, complete footprint"
    }
    negative_texts = ["close-up detail", "partial view", "blurred", "text overlay", "night view", "human selfie"]
    results = {}

    try:
        img_embs = CLIP_MODEL.encode(valid_images)
        neg_embs = CLIP_MODEL.encode(negative_texts)
        neg_sim_matrix = util.cos_sim(img_embs, neg_embs)
        neg_penalties = neg_sim_matrix.max(dim=1).values.numpy()

        for view_name, prompt in view_prompts.items():
            text_emb = CLIP_MODEL.encode([prompt])
            pos_scores = util.cos_sim(text_emb, img_embs)[0].numpy()
            best_score = -999.0
            best_idx = -1

            for i in range(len(valid_images)):
                base_score = pos_scores[i] - (0.35 * neg_penalties[i])
                final_score = base_score
                img = valid_images[i]
                w, h = img.size
                ratio = w / h

                if view_name == "front" and (ratio < 0.4 or ratio > 2.5):
                    final_score -= 0.15
                elif view_name == "top_down" and ratio > 0.8:
                    final_score += 0.05

                if final_score > best_score:
                    best_score = final_score
                    best_idx = i

            if best_idx != -1 and pos_scores[best_idx] > 0.25:
                results[view_name] = valid_urls[best_idx]
                print(f"✅ [View: {view_name}] Selected (Sim: {pos_scores[best_idx]:.3f})")
            else:
                if best_idx != -1:
                    print(f"🗑️ [View: {view_name}] Rejected (Best Sim {pos_scores[best_idx]:.3f} < 0.25)")

        return results
    except Exception as e:
        print(f"⚠️ Multi-view selection error: {e}")
        return {}


def _analyze_image_semantics(image_url: str, entity_name: str) -> Dict[str, str]:
    if not image_url: return {}
    print(f"👁️ [VLM] 正在对选中图片进行视觉结构分析: {entity_name} ...")
    base64_image = ""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(image_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            base64_image = base64.b64encode(resp.content).decode('utf-8')
        else:
            return {}
    except Exception:
        return {}

    prompt = f"""
        You are an expert cartographer analyzing the landmark '{entity_name}'.
        CRITICAL INSTRUCTION - KNOWLEDGE FIRST:
        1. First, RECALL your internal knowledge about this landmark's true 3D topology. 
        2. Then, analyze the image.
        3. DETECT ILLUSIONS: If the image view hides the true shape, you MUST report the TRUE 3D SHAPE based on your knowledge, not just the 2D projection.

        4. GEOMETRIC PRIMITIVES MANDATORY: You MUST describe the structure using explicit geometric shapes (e.g., 'bulbous dome', 'fat cylindrical belly', 'tapered conical spire', 'cubic base'). DO NOT rely on generic cultural terms like 'pagoda' or 'temple' without describing their exact geometry.

        Return JSON only:
        {{
            "posture": "standing|sitting|reclining(lying_down)|abstract|ring_shaped|crossing_arches", 
            "orientation": "vertical(tall)|horizontal(wide)|square",
            "shape_description": "A precise GEOMETRIC description of the TRUE 3D structural composition (e.g., 'A solid bulbous hemispherical base topped with a stepped conical spire')."
        }}
        """
    try:
        response = client.chat.completions.create(
            model=MODELS["LLM_MODEL"],
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}},
                ]},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return extract_json(response.choices[0].message.content) or {}
    except Exception:
        return {}


def _search_baidu_image(keyword: str) -> Optional[str]:
    cands = _search_baidu_candidates(keyword, limit=5)
    return cands[0] if cands else None


def _gather_raw_knowledge(user_text: str, search_focus: str = None) -> Tuple[str, Optional[str]]:
    queries = _expand_queries(user_text)
    blobs = []
    first_image = None
    if search_focus: queries.insert(0, search_focus)
    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in user_text)

    for q in queries:
        if has_chinese:
            summary, img = _fetch_baidu_baike(q)
            if summary:
                blobs.append(f"[Baidu] {q}\n{summary}")
                if not first_image and img: first_image = img
                continue
        langs = _langs_for(q, user_text)
        for lang in langs:
            title = _wiki_search(q, lang)
            if title:
                data = _wiki_summary(title, lang)
                if data:
                    extract = data.get("extract")
                    img_src = data.get("thumbnail", {}).get("source") or data.get("originalimage", {}).get("source")
                    if extract:
                        blobs.append(f"[Wiki-{lang}] {title}\n{extract}")
                        if not first_image and img_src: first_image = img_src
                        break
    text = "\n\n".join(blobs)
    return text, first_image


def _fetch_baidu_baike(keyword: str) -> Tuple[Optional[str], Optional[str]]:
    url = f"https://baike.baidu.com/item/{keyword}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
        if resp.status_code != 200: return None, None
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')
        texts = []
        summary_div = soup.find('div', class_='lemma-summary')
        if summary_div: texts.append(summary_div.get_text().strip())
        summary_text = "\n".join(texts)
        if not summary_text: return None, None

        image_url = None
        meta_img = soup.find('meta', property="og:image")
        if meta_img: image_url = meta_img.get("content")
        if image_url and image_url.startswith('//'): image_url = "https:" + image_url
        return summary_text, image_url
    except Exception:
        return None, None


def _wiki_search(q: str, lang="en") -> Optional[str]:
    try:
        params = {"action": "opensearch", "search": q, "limit": 1, "namespace": 0, "format": "json"}
        r = requests.get(WIKI_SEARCH.format(lang=lang), params=params, timeout=5)
        if r.status_code == 200:
            j = r.json()
            if isinstance(j, list) and len(j) >= 2 and j[1]: return j[1][0]
    except Exception:
        pass
    return None


def _wiki_summary(title: str, lang="en") -> Optional[Dict[str, Any]]:
    try:
        url = WIKI_SUMMARY.format(lang=lang, title=title.replace(" ", "_"))
        r = requests.get(url, timeout=5, headers={"accept": "application/json"})
        if r.status_code == 200: return r.json()
    except Exception:
        pass
    return None


def _expand_queries(user_text: str) -> List[str]:
    qs: List[str] = [user_text.strip()]
    for seg in re.findall(r"[一-龥A-Za-z0-9·\-\s]{2,}", user_text):
        s = seg.strip()
        if s and s not in qs: qs.append(s)
    return list(dict.fromkeys(qs))


def _langs_for(q: str, user_text: str) -> List[str]:
    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in user_text + q)
    return ["zh", "en"] if has_chinese else ["en", "zh"]


# ==============================================================================
# 🧠 [ReAct Tools Schema & Agent Orchestrator]
# ==============================================================================

GROUNDER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "gather_text_knowledge",
            "description": "Gather text descriptions from encyclopedias (Baidu/Wiki). Useful for finding basic attributes.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_and_analyze_multi_view_images",
            "description": "Search the web for multi-view images of the landmark and use VLM to extract its TRUE 3D structure.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    }
]


def ground_entity_to_spec(user_text: str, search_focus: str = None) -> Dict[str, Any]:
    """
    [ReAct Agent 主控] 使用 OpenAI Tools Calling 实现的受限智能体调研循环。
    最多思考 3 步，防止陷入死循环，兼顾鲁棒性与系统稳定性。
    """
    base_query = search_focus if search_focus else user_text
    print(f"🚀 [Research Agent] 启动自主调研循环 (Target: {base_query})")

    system_prompt = (
        "You are a Research Agent for Cartography. Your goal is to find the exact 3D physical structure of a landmark.\n"
        "You MUST use tools to gather both text and visual evidence. \n"
        "If you have enough information to fill the JSON schema, DO NOT call tools anymore. Output the final JSON directly.\n"
        "Schema:\n"
        "{ \n"
        "  \"entity\": {\"name\": str, \"location\": str},\n"
        "  \"entity_type\": \"bridge|tower|building|statue|logogram|other\",\n"
        "  \"structure\": {\n"
        "      \"structural_system\": \"truss|arch|suspension|beam|unknown\",\n"
        "      \"shape_features\": [str],  // MUST use explicit geometric shapes (e.g., 'fat bulbous belly', 'hemispherical dome', 'stepped pyramid base'). AVOID generic cultural labels.\n"
        "      \"material\": \"steel|stone|concrete|wood\",\n"
        "      \"view_recommendation\": \"side|front|isometric|top_down\"\n"
        "  },\n"
        "  \"constraints\": {\n"
        "      \"must\": [str],      // Explicit visual GEOMETRIC elements that MUST appear in the final design (e.g., 'The base MUST be a fat round bulbous dome').\n"
        "      \"must_not\": [str]   // Elements to EXCLUDE (e.g., 'must not be a straight multi-tier pavilion pagoda').\n"
        "  }\n"
        "}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Please investigate this landmark and output the structural JSON spec: {base_query}"}
    ]

    final_spec = {"entity": {"name": base_query}, "constraints": {"must_not": []}}
    accumulated_visual_pack = {}
    fallback_image = None

    # 限制最大思考步数，作为“防沉迷”锁
    MAX_STEPS = 3

    for step in range(MAX_STEPS):
        print(f"   🤔 [ReAct] 思考步数 {step + 1}/{MAX_STEPS}...")
        resp = client.chat.completions.create(
            model=MODELS["LLM_MODEL"],
            messages=messages,
            tools=GROUNDER_TOOLS,
            tool_choice="auto"
        )
        msg = resp.choices[0].message
        messages.append(msg)

        # 1. 如果大模型决定调用工具
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"   🛠️ [Tool Calling] 决定使用工具: {func_name}({args})")

                tool_result = ""
                if func_name == "gather_text_knowledge":
                    text, img = _gather_raw_knowledge(args["query"])
                    if img and not fallback_image:
                        fallback_image = img
                    tool_result = text if text else "No text descriptions found."

                elif func_name == "search_and_analyze_multi_view_images":
                    # 执行强化的视觉搜图和匹配
                    q = args["query"]
                    robust_queries = [q, f"{q} aerial top down view", f"{q} isometric structure"]
                    all_cands = []
                    for rq in robust_queries:
                        all_cands.extend(_search_baidu_candidates(rq, limit=10))

                    visual_pack = _multi_view_clip_selection(q, list(set(all_cands)))

                    if visual_pack:
                        accumulated_visual_pack.update(visual_pack)
                        target_url = visual_pack.get("top_down") or visual_pack.get("isometric") or visual_pack.get(
                            "front") or list(visual_pack.values())[0]
                        vlm_traits = _analyze_image_semantics(target_url, q)
                        tool_result = f"Found Multi-View Images: {list(visual_pack.keys())}. \nVLM Analysis of Structure: {vlm_traits}"
                    else:
                        tool_result = "Failed to find clear multi-view images."

                # 将工具执行结果追加回上下文
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })

        # 2. 如果大模型认为信息足够，直接输出 JSON 结果
        else:
            print("   ✅ [ReAct] 证据收集完毕，已生成结构 Spec。")
            extracted = extract_json(msg.content)
            if extracted: final_spec = extracted
            break

    # 3. 兜底逻辑：如果达到最大步数，强制输出 JSON
    last_msg = messages[-1]
    is_tool = (isinstance(last_msg, dict) and last_msg.get("role") == "tool") or \
              (hasattr(last_msg, "role") and last_msg.role == "tool")

    if step == MAX_STEPS - 1 and is_tool:
        print("   ⚠️ [ReAct] 达到最大步数，强制总结输出。")
        messages.append({"role": "user",
                         "content": "Max steps reached. You MUST output the final JSON spec now based on the gathered information."})
        resp = client.chat.completions.create(
            model=MODELS["LLM_MODEL"],
            messages=messages,
            response_format={"type": "json_object"}
        )
        extracted = extract_json(resp.choices[0].message.content)
        if extracted: final_spec = extracted

    # 4. 收尾：把记忆中的图片 URL 塞进最终结果，供 Orchestrator 和 Reviewer 使用
    if accumulated_visual_pack:
        final_spec["reference_images"] = accumulated_visual_pack
        final_spec["reference_image_url"] = accumulated_visual_pack.get("isometric") or accumulated_visual_pack.get(
            "top_down") or accumulated_visual_pack.get("front") or list(accumulated_visual_pack.values())[0]
        # 保存 VLM 事实
        final_spec["vlm_analysis"] = extract_json(str(tool_result)) if 'vlm_traits' in locals() else {}
    elif fallback_image:
        final_spec["reference_image_url"] = fallback_image
        final_spec["reference_images"] = {"front": fallback_image}

    save_json("Grounder_spec", final_spec)
    return final_spec