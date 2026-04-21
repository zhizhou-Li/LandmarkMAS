# -*- coding: utf-8 -*-
# SymbolGeneration/Agent/agents/interpreter_agent.py
from openai import OpenAI
from config import MODELS, OPENAI_API_KEY
from utils import log

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM = (
    "You convert a Chinese/English user request into a COMPACT JSON intent schema. "
    "Return ONLY JSON with these keys (optional unless obvious):\n"
    "{"
    ' "entity": {"name": str, "geography": str},'
    ' "entity_type": "bridge|tower|building|gate|monument|pagoda|church|mosque|temple|statue|other",'
    ' "superstructure": "truss|arch|suspension|cable_stayed|beam|frame|dome|unknown",'
    ' "material_hint": "steel|stone|concrete|wood|mixed|unknown",'
    ' "style_intent": "illustration|logo|poster|render|other",'
    ' "background": "transparent|white|none|unspecified"'
    "}\n"
    "- If not clear, prefer high-level guesses (e.g., superstructure='unknown')."
)

def run_interpreter(user_text: str) -> str:
    resp = client.chat.completions.create(
        model=MODELS["LLM_MODEL"],
        temperature=0.0,
        top_p=1,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":f"User request:\n{user_text}\nReturn JSON only."}
        ]
    )
    content = resp.choices[0].message.content
    log("CommandInterpreter", content)
    return content
