# utils.py（替换）
import os, time, json, re
from pathlib import Path

# 把输出目录固定为 utils.py 所在目录下的 outputs/
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def log(agent_name, content):
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = OUTPUT_DIR / f"{agent_name}_{ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content if isinstance(content, str) else str(content))
    print(f"✅ [{agent_name}] 输出已保存到 {path}")

def save_json(agent_name, data):
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = OUTPUT_DIR / f"{agent_name}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ [{agent_name}] JSON结果已保存到 {path}")

def extract_json(text: str):
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{[\s\S]*\}', text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}
