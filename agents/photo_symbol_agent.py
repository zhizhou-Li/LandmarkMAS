# -*- coding: utf-8 -*-
# SymbolGeneration/Agent/agents/photo_symbol_agent.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np
from openai import OpenAI

from config import OPENAI_API_KEY, MODELS
from utils import log, save_json
from .prompt_planner import compile_prompt
from .grounder_agent import ground_entity_to_spec
from .spec_infer_agent import infer_structure_spec
from .spec_utils import merge_specs
from .detector_agent import run_detector
from .generator_agent import run_generator

# 若你已添加 vectorizer_agent.py，则可启用 SVG 导出
try:
    from .vectorizer_agent import png_to_svg
except Exception:
    png_to_svg = None  # 没有矢量化依赖也可以先跑 PNG

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------- 1) 轮廓/蒙版 ----------
def _largest_component_mask(img_bgr: np.ndarray, pad: int = 12) -> np.ndarray:
    """取最大连通域作为主体，返回 0/255 mask（白=主体）"""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 25, 7)
    num, labels = cv2.connectedComponents(thr)
    if num <= 1:
        return np.zeros((h,w), dtype=np.uint8)
    areas = [(labels==i).sum() for i in range(1, num)]
    cid = 1 + int(np.argmax(areas))
    mask = (labels==cid).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.dilate(mask, k, iterations=1)
    return mask


def build_silhouette_and_mask(image_path: str, out_dir: Path) -> Tuple[str, str]:
    """
    输出：
      - silhouette_path: 边缘线稿 PNG（参考）
      - mask_path:       RGBA PNG；alpha=0 的区域会被编辑（可绘制），alpha=255 保持不变
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]

    # 边缘线稿（可选参考）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    silu = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 最大连通域当做“主体”
    fg = _largest_component_mask(img)  # 255 = 主体，0 = 背景

    # ★ 关键：主体设为透明（0，表示可编辑），背景设为不透明（255，表示锁定）
    alpha = np.where(fg == 255, 0, 255).astype(np.uint8)
    rgba  = np.dstack([np.zeros((h, w, 3), dtype=np.uint8), alpha])

    out_dir.mkdir(parents=True, exist_ok=True)
    silhouette_path = str(out_dir / "photo_silhouette.png")
    mask_path       = str(out_dir / "photo_mask.png")
    cv2.imwrite(silhouette_path, silu)
    cv2.imwrite(mask_path, rgba)
    log("Photo2Symbol_Mask", f"silhouette={silhouette_path}\nmask={mask_path}")
    return silhouette_path, mask_path


# ---------- 2) 二色调提取（KMeans，缺失则用均值兜底） ----------
def extract_two_tone_palette(image_path: str) -> List[str]:
    try:
        from sklearn.cluster import KMeans
    except Exception:
        img = cv2.imread(image_path)
        if img is None:
            return ["#223344", "#99AAC0"]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1,3)
        mean = rgb.mean(axis=0).astype(int)
        dark = np.clip(mean - 40, 0, 255)
        light = np.clip(mean + 40, 0, 255)
        to_hex = lambda arr: f"#{arr[0]:02X}{arr[1]:02X}{arr[2]:02X}"
        return sorted([to_hex(dark), to_hex(light)],
                      key=lambda h: int(h[1:3],16)*0.3 + int(h[3:5],16)*0.59 + int(h[5:7],16)*0.11)

    img = cv2.imread(image_path)
    if img is None:
        return ["#223344", "#99AAC0"]
    data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1,3).astype(np.float32)
    km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(data)
    centers = km.cluster_centers_.astype(int)
    hexes = [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in centers]
    # 深浅排序
    hexes.sort(key=lambda h: int(h[1:3],16)*0.3 + int(h[3:5],16)*0.59 + int(h[5:7],16)*0.11)
    return hexes


# ---------- 3) 主入口：实景 → 符号 ----------
def photo_to_symbol(
    image_path: str,
    user_text: str,
    user_structure_spec: Optional[Dict[str, Any]] = None,
    use_edits_first: bool = True,
    export_svg: bool = True,
) -> Dict[str, Any]:
    """
    高层封装：给实景图和需求，返回本地 PNG & 可选 SVG。
    不改你原有 orchestrator；需要时直接调用本函数即可。
    """
    OUT_DIR = (Path(__file__).resolve().parents[1] / "outputs" / "photo2symbol")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # a) 视觉检测（名称/要素）
    schema = '{"kind":"landmark"}'  # 轻量占位；可替换为 run_interpreter(user_text)
    det = run_detector(image_path, schema)

    # b) Grounder + 推断 + 合并
    grounded = ground_entity_to_spec(user_text)
    auto = infer_structure_spec(user_text=user_text, detector_spec=det or None)
    merged = merge_specs(user_spec=user_structure_spec or auto, detector_spec=det, defaults=grounded)

    # c) 蒙版/轮廓 + 二色调样式
    silhouette_path, mask_path = build_silhouette_and_mask(image_path, OUT_DIR)
    palette = extract_two_tone_palette(image_path)
    style_json = json.dumps({
        "style_name": "Photo2Symbol_TwoTone",
        "stroke": {"width": 3, "pattern": "solid", "corner": "round"},
        "fill": {"type": "flat", "opacity": 1.0},
        "palette": palette,
        "simplification": {"tolerance_px": 2, "max_points": 120},
        "iconography": {"emphasis": ["outline"], "negative_space": True},
        "export": {"size": 512, "background": "transparent"}
    }, ensure_ascii=False)

    # d) 生成：优先蒙版编辑，失败回退纯生成
    result_paths = run_generator(
        outline_path=silhouette_path,
        style_json=style_json,
        user_text=user_text,
        structure_spec=merged,
        base_image=image_path if use_edits_first else None,
        mask_image=mask_path if use_edits_first else None
    )
    best_png = result_paths[0]

    # e) 可选：转 SVG
    svg_path = None
    if export_svg and png_to_svg is not None:
        try:
            svg_path = png_to_svg(best_png, method="auto", threshold=180, simplify_eps=1.0)
        except Exception as e:
            log("Photo2Symbol_SVG", f"svg failed: {e}")

    info = {"png": best_png, "svg": svg_path, "palette": palette, "structure": merged}
    save_json("Photo2Symbol_summary", info)
    return info
