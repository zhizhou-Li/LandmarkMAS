# -*- coding: utf-8 -*-
# SymbolGeneration/Agent/agents/geometry_descriptor_agent.py
from __future__ import annotations
import re, json
from typing import Dict, Any, List
from openai import OpenAI
from config import OPENAI_API_KEY, MODELS
from utils import log, extract_json, save_json

client = OpenAI(api_key=OPENAI_API_KEY)

SURFACE_TO_SYSTEM = [
    (r"\btruss\b|桁架|桁梁|网架", "truss"),
    (r"\barch\b|拱桥(?!形上弦)|拱结构", "arch"),
    (r"\bsuspension\b|悬索|吊桥", "suspension"),
    (r"cable[-\s]?stayed|斜拉", "cable_stayed"),
    (r"\bbeam\b|梁桥", "beam"),
    (r"\bframe\b|框架", "frame"),
]
SURFACE_TO_TOPCHORD = [
    (r"camelback|帕克|多折|分段拱|弓形上弦|拱形上弦|弧形上弦", "polygonal"),
]
SURFACE_TO_MAT = [
    (r"\bsteel\b|钢", "steel"),
    (r"\bstone\b|石", "stone"),
    (r"\bconcrete\b|混凝土", "concrete"),
    (r"\bwood\b|木", "wood"),
]

SYSTEM = (
    "You are a geometry descriptor module. Distinguish structural system from visual top-chord shape. "
    "Output ONLY JSON with:\n"
    "{"
    ' "entity_type": "bridge|tower|building|gate|monument|pagoda|other",'
    ' "structural_system": "truss|arch|suspension|cable_stayed|beam|frame|unknown",'
    ' "top_chord_profile": "flat|polygonal|camelback|curved|unknown",'
    ' "arch_rib_presence": true|false,'
    ' "material_hint": "steel|stone|concrete|wood|mixed|unknown",'
    ' "spans": int, "piers": int,'
    ' "view": "side_elevation|isometric|front|perspective|emblematic|unknown",'
    ' "composition": {"centered": bool, "full_silhouette": bool, "balanced_negative_space": bool},'
    ' "constraints": {"must":[str], "must_not":[str]}'
    "}\n"
    "- If structural_system='truss' but visual text mentions arch-like top chord, set top_chord_profile='polygonal' and arch_rib_presence=false."
)

def _regex_norm(value: str, pairs: List) -> str|None:
    if not value: return None
    txt = value.lower()
    for pat, norm in pairs:
        if re.search(pat, txt):
            return norm
    return None

def describe_geometry(user_text: str, grounded_blob: str = "") -> Dict[str, Any]:
    prefill: Dict[str, Any] = {}
    txt = (user_text or "") + "\n" + (grounded_blob or "")
    ss = _regex_norm(txt, SURFACE_TO_SYSTEM)
    tc = _regex_norm(txt, SURFACE_TO_TOPCHORD)
    mt = _regex_norm(txt, SURFACE_TO_MAT)
    if ss: prefill["structural_system"] = ss
    if tc: prefill["top_chord_profile"] = tc
    if mt: prefill["material_hint"] = mt

    msgs = [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":[
            {"type":"text","text":f"User text:\n{user_text}"},
            {"type":"text","text":f"Grounded snippets:\n{grounded_blob or '(none)'}"},
            {"type":"text","text":"Bridge icon preferred; use side_elevation or isometric."}
        ]}
    ]
    resp = client.chat.completions.create(
        model=MODELS["LLM_MODEL"],
        temperature=0.0,
        response_format={"type":"json_object"},
        messages=msgs
    )
    raw = resp.choices[0].message.content
    log("GeometryDescriptor_raw", raw)
    data = extract_json(raw) or {}

    for k,v in prefill.items():
        data.setdefault(k, v)

    comp = data.get("composition") or {}
    comp.setdefault("centered", True)
    comp.setdefault("full_silhouette", True)
    comp.setdefault("balanced_negative_space", True)
    data["composition"] = comp

    # 当为桁架体系时，默认没有连续拱肋
    if data.get("structural_system") == "truss":
        data["arch_rib_presence"] = False

    mn = set((data.get("constraints") or {}).get("must_not") or [])
    mn.update(["suspension towers","cables","stone arches","single big arch","continuous arch rib"])
    data["constraints"] = {
        "must": list((data.get("constraints") or {}).get("must") or []),
        "must_not": sorted(mn)
    }

    save_json("GeometryDescriptor", data)
    return data
