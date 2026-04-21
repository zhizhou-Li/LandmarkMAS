# -*- coding: utf-8 -*-
# SymbolGeneration/Agent/agents/prompt_planner.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple
from .spec_utils import json_to_constraints

def _parse_json(obj: Any) -> Dict[str, Any]:
    if obj is None: return {}
    if isinstance(obj, dict): return obj
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("{") and s.endswith("}"):
            try: return json.loads(s)
            except Exception: return {}
    return {}

HINTS = {
    "engineering": ["bridge","tower","pagoda","gate","cathedral","building","结构","几何","桥","塔","建筑"],
    "art": ["illustration","poster","risograph","watercolor","插画","版画","海报","艺术"],
    "concept": ["sci-fi","fantasy","未来","科幻","概念"],
    "product": ["product","render","材质","产品","渲染"],
    "logo": ["logo","icon","glyph","mark","徽标","图标","矢量"]
}

SCHEMAS = {
    "engineering": ["scene","geometry","style","composition","negative","style_json"],
    "art":         ["scene","color_mood","style","composition","texture_detail","negative","style_json"],
    "concept":     ["concept_core","environment_material","tech_epoch","lighting_mood","negative","style_json"],
    "product":     ["scene","material_detail","lighting","composition","negative","style_json"],
    "logo":        ["scene","shape_language","minimalism","negative","style_json"]
}

def _guess_task(user_text: str, spec: Dict[str,Any]) -> str:
    if isinstance(spec.get("task_type"), str) and spec["task_type"]:
        return spec["task_type"]
    txt = (user_text + " " + json.dumps(spec, ensure_ascii=False)).lower()
    for t, keys in HINTS.items():
        if any(k in txt for k in keys): return t
    return "art"

def _constraints_from_spec(spec: Dict[str,Any]) -> Tuple[List[str], List[str]]:
    return json_to_constraints(spec)

# ---------------- Statue auto reasoning (NEW) ----------------
def _is_statue_context(user_text: str, spec: Dict[str, Any]) -> bool:
    if (spec.get("entity_type") or "").lower() == "statue":
        return True
    t = (user_text + " " + json.dumps(spec, ensure_ascii=False)).lower()
    return ("雕像" in t) or ("雕塑" in t) or ("statue" in t) or ("sculpture" in t)

def _detect_statue_archetype(user_text: str, spec: Dict[str, Any]) -> str:
    """
    轻量原型识别：不联网、零依赖；只根据用户文本 + entity 命名做启发式判别。
    返回: 'mother_child' | 'liberty' | 'equestrian' | 'seated_buddha' | 'generic'
    """
    txt = (user_text or "").lower()
    ent = spec.get("entity") or {}
    name_blob = " ".join([
        str(ent.get("name") or ""),
        *[str(x) for x in (ent.get("aliases") or [])],
        str(ent.get("location") or "")
    ]).lower()
    blob = txt + " " + name_blob

    # 自由女神像
    liberty_keys = ("自由女神", "statue of liberty", "liberty enlightenment", "nyc", "new york", "torch", "七芒", "seven rays", "crown")
    if any(k in blob for k in liberty_keys):
        return "liberty"

    # 兰州黄河母亲 / 母子像
    mother_keys = ("黄河母亲", "母亲", "母子", "mother and child", "mother-child")
    if any(k in blob for k in mother_keys):
        return "mother_child"

    # 骑马像
    eq_keys = ("骑马", "马", "equestrian", "horseback")
    if any(k in blob for k in eq_keys):
        return "equestrian"

    # 坐佛/佛像
    buddha_keys = ("佛", "buddha", "坐佛", "释迦", "观音", "guanyin")
    if any(k in blob for k in buddha_keys):
        return "seated_buddha"

    return "generic"
# ------------------------------------------------------------

def _slot_texts(task: str) -> Dict[str,str]:
    if task == "engineering":
        return {
            "scene": "Subject: clear landmark depiction; emblematic or side-elevation for recognizability.",
            "geometry": "Emphasize structural consistency and repeating spans; coherent proportions.",
            "style": "Architectural/engineering illustration; fine linework; muted limited palette; flat shading.",
            "composition": "Centered; full silhouette; background hints allowed but not dominant.",
            "negative": "No people, no vehicles, no random ads."
        }
    if task == "art":
        return {
            "scene": "Subject: artistic micro-map icon.",
            "color_mood": "Muted/duotone palette with high contrast.",
            "style": "Poster-like; clean silhouette; minimal noise.",
            "composition": "Balanced negative space; strong figure-ground separation.",
            "texture_detail": "Subtle paper grain; avoid cluttered micro-details.",
            "negative": "No photorealism; no tiny text."
        }
    if task == "product":
        return {
            "scene": "Subject centered; orthographic-friendly view.",
            "material_detail": "Crisp material depiction; clean edges.",
            "lighting": "Soft key light; avoid harsh glare.",
            "composition": "Neutral background; strong readability.",
            "negative": "No distracting props."
        }
    return {
        "scene": "Subject distilled as a logo/icon.",
        "shape_language": "Geometric reduction; strong silhouette; few primitives.",
        "minimalism": "Limited palette; avoid unnecessary gradients.",
        "negative": "No text; avoid photorealism."
    }

def compile_prompt(user_text: str,
                   style_json: str,
                   structure_spec: Dict[str, Any] | str | None = None,
                   variation_note: str = "") -> str:
    """
    根据用户文本 + 多源结构约束，生成最终喂给图像模型的提示词。

    注意：
    - json_to_constraints(structure_spec) 在当前工程中返回的是 (must, must_not) 元组。
      所以这里显式拆包，不再使用 `.must` 这样的属性访问，避免 'tuple' 报错。
    - baseline 调用时 structure_spec=None => must/must_not 为空列表，只用通用说明。
    - multi-agent 调用时会有 grounder/spec_infer 注入的结构和 must/must_not，从而形成差异。
    """
    # 解析结构 spec，确保是 dict
    spec = _parse_json(structure_spec)
    # 从 spec 中提取 must / must_not 约束（项目里这个函数返回 tuple）
    must, must_not = json_to_constraints(spec)

    # --- 1. 动态构建主体描述 (Generic Subject Description) ---
    ent = spec.get("entity", {})
    ent_name = ent.get("name") or user_text
    ent_type = spec.get("entity_type", "landmark")

    # 获取结构细节
    struct = spec.get("structure", {})
    sys_type = struct.get("structural_system") or struct.get("superstructure")
    mat_hint = struct.get("material") or struct.get("material_hint")
    view_hint = struct.get("view_recommendation") or spec.get("view")

    parts: List[str] = []

    # ---------- 基础符号风格指令（所有方法共享） ----------
    parts.append(
        "You are generating a cartographic POI symbol for a digital map. "
        "The output must be a clean 2D icon, not a full illustration or photo."
    )
    parts.append(
        "Style: strong silhouette, minimal lines, uniform stroke width, "
        "black or two-tone line art on a flat or transparent background, "
        "suitable for small-size display."
    )

    # ---------- 动态主体描述 ----------
    subject_desc = f"Subject: '{ent_name}'."
    if ent_type and ent_type != "other":
        subject_desc += f" Type: {ent_type}."

    # 注入结构特征
    features = []
    if sys_type and sys_type != "unknown":
        features.append(f"Structure: {sys_type}")
    if mat_hint and mat_hint != "unknown":
        features.append(f"Material: {mat_hint}")
    if view_hint and view_hint != "unknown":
        features.append(f"View: {view_hint}")

    # 注入 shape_features (来自 Grounder 的通用提取)
    shape_feats = struct.get("shape_features", [])
    if isinstance(shape_feats, list):
        features.extend([str(f) for f in shape_feats if f])

    if features:
        subject_desc += " (" + "; ".join(features) + ")."

    parts.append(subject_desc)

# ---------- 用户原始意图 ----------
    if user_text:
        parts.append(f"User context: {user_text}")

    # ---------- 强约束注入 (完全依赖 RAG 提取结果) ----------
    # 不再根据关键词 if/else，而是直接把 spec 里的 must/must_not 填进去
    if must:
        # dict.fromkeys 去重
        reqs = "; ".join(dict.fromkeys(str(m) for m in must if m))
        parts.append(f"Hard requirements (MUST include these visual cues): {reqs}")

    if must_not:
        forbids = "; ".join(dict.fromkeys(str(m) for m in must_not if m))
        parts.append(f"Forbidden elements (do NOT include): {forbids}")

    # ---------- 样式控制 (Designer 输出) ----------
    sj = (style_json or "").strip()
    if sj and sj not in ("{}", "[]"):
        parts.append(
            "Follow this JSON style configuration for stroke/palette/composition "
            "(do not render the JSON text itself):"
        )
        parts.append(sj)

    # ---------- 多轮迭代时的变体指示 ----------
    if variation_note:
        parts.append(variation_note)

    # ---------- 全局负向约束 ----------
    parts.append(
        "General negative: no photographic texture, no illegible small text, "
        "no cluttered background. Keep the landmark symbol bold, centered and highly recognizable."
    )

    return "\n".join(p for p in parts if p)

