# -*- coding: utf-8 -*-
from __future__ import annotations
import json, re
from typing import Any, Dict, List, Tuple


def _parse_json(obj: Any) -> Dict[str, Any]:
    if obj is None: return {}
    if isinstance(obj, dict): return obj
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except Exception:
                return {}
    return {}


def merge_specs(defaults: Dict[str, Any] = None,
                user_spec: Dict[str, Any] = None,
                detector_spec: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    合并多源约束。
    核心策略：Detector (视觉) > User (用户) > Defaults (Grounder/SpecInfer)
    视觉看到的结构特征具有最高优先级。
    """
    if defaults is None: defaults = {}
    if user_spec is None: user_spec = {}
    if detector_spec is None: detector_spec = {}

    # 1. 以 Grounder/SpecInfer 的结果为底板
    final_spec = defaults.copy()

    # 辅助：递归合并字典
    def safe_merge(target, source):
        for k, v in source.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                safe_merge(target[k], v)
            elif isinstance(v, list) and k in target and isinstance(target[k], list):
                # 列表合并：去重并保留顺序
                target[k] = list(dict.fromkeys(target[k] + v))
            elif v is not None and v != "" and v != "unknown":
                target[k] = v

    # 2. 合并用户输入 (User Spec) - 用户的明确指令通常需要保留
    # 但我们这里只是做基础合并，稍后用 Detector 覆盖关键结构
    safe_merge(final_spec, user_spec)

    # 3. [关键] 合并 Detector Spec (视觉证据)
    # 视觉侦察兵看到的信息（如姿态、构图）必须覆盖文本推断的幻觉
    if detector_spec:
        # (A) 强行覆盖关键几何字段
        if "structure" in detector_spec:
            det_struct = detector_spec["structure"]
            if "structure" not in final_spec: final_spec["structure"] = {}
            target_struct = final_spec["structure"]

            # 这些字段如果视觉看到了，绝对以视觉为准
            priority_fields = ["composition", "posture", "dominant_lines", "view_recommendation", "structural_system"]
            for field in priority_fields:
                if field in det_struct and det_struct[field] and det_struct[field] != "unknown":
                    target_struct[field] = det_struct[field]

            # (B) 合并 shape_features，把视觉特征插到最前面！
            if "shape_features" in det_struct:
                visual_feats = det_struct["shape_features"]
                if isinstance(visual_feats, list):
                    existing = target_struct.get("shape_features", [])
                    if isinstance(existing, str): existing = [existing]
                    # 视觉特征 + 文本特征 (去重)
                    target_struct["shape_features"] = list(dict.fromkeys(visual_feats + existing))

        # (C) 合并约束 (Must/Must Not)
        if "constraints" in detector_spec:
            target_cons = final_spec.setdefault("constraints", {})
            det_cons = detector_spec["constraints"]

            # Must: 视觉看到的必须画
            if "must" in det_cons:
                existing_must = target_cons.get("must", [])
                target_cons["must"] = list(dict.fromkeys(det_cons["must"] + existing_must))

            # Must Not: 视觉说没有的就别画
            if "must_not" in det_cons:
                existing_not = target_cons.get("must_not", [])
                target_cons["must_not"] = list(dict.fromkeys(det_cons["must_not"] + existing_not))

        # (D) 补充 entity_type 如果缺失
        if "entity_type" in detector_spec and detector_spec["entity_type"] != "unknown":
            final_spec["entity_type"] = detector_spec["entity_type"]

    return final_spec


def json_to_constraints(spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """将 Spec 转换为 Prompt 用的 (MUST[], MUST_NOT[]) 列表"""
    must: List[str] = []
    must_not: List[str] = []

    if not spec:
        return must, must_not

    # 1. 提取 constraints 里的显式约束
    cons = spec.get("constraints", {})
    if isinstance(cons.get("must"), list):
        must.extend(str(x) for x in cons["must"])
    if isinstance(cons.get("must_not"), list):
        must_not.extend(str(x) for x in cons["must_not"])

    # 2. 提取 structure 里的关键特征
    struct = spec.get("structure", {})
    # 如果有 shape_features，也加入 must
    if "shape_features" in struct and isinstance(struct["shape_features"], list):
        must.extend(str(x) for x in struct["shape_features"])

    return list(dict.fromkeys(must)), list(dict.fromkeys(must_not))


# ---- 规范化 ----
def normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        return {}

    # 1. 确保结构字段存在
    if "structure" not in spec: spec["structure"] = {}
    if "constraints" not in spec: spec["constraints"] = {}

    # 2. 确保 shape_features 是列表 (防止 LLM 输出字符串)
    struct = spec["structure"]
    if "shape_features" in struct:
        if isinstance(struct["shape_features"], str):
            struct["shape_features"] = [struct["shape_features"]]
        elif not isinstance(struct["shape_features"], list):
            struct["shape_features"] = []

    # 3. 桥梁特有字段清理 (防止污染非桥梁实体)
    et = (spec.get("entity_type") or "").lower()
    if et and "bridge" not in et:
        for k in ("top_chord_profile", "arch_rib_presence", "spans", "piers"):
            struct.pop(k, None)

        # 清理 must_not 里的桥梁术语
        cons = spec.get("constraints", {})
        if "must_not" in cons and isinstance(cons["must_not"], list):
            cons["must_not"] = [
                x for x in cons["must_not"]
                if not any(t in str(x).lower() for t in ("suspension cables", "truss nodes"))
            ]

    return spec