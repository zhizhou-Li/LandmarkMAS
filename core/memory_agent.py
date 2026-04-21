# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from typing import Dict, Any, Optional

from .agent_base import Agent
from .messages import Msg, TOPICS

def _entity_key(merged_spec: Dict[str, Any]) -> str:
    ent = (merged_spec or {}).get("entity") or {}
    # 以 name + location 组成 key，尽量稳定
    name = str(ent.get("name") or "").strip().lower()
    loc  = str(ent.get("location") or "").strip().lower()
    return f"{name}|{loc}" if (name or loc) else "unknown"

class MemoryAgent(Agent):
    """
    轻量“长期记忆”：
    - 监听 MERGE_RES 获取实体 key（job -> entity_key）
    - 监听 DESIGN_RES 暂存当前 style_json（job -> style）
    - 监听 ARBITER_RES 若决策 stop，则把该 job 的 style 记为该实体的“最佳样式”
    - 监听 VECTOR_RES 记录最终 SVG 路径
    说明：不额外定义 memory.query 主题；Planner/Designer 如需复用，直接用 bb.mem_get(...)
    """
    def __init__(self, bb):
        super().__init__("MemoryAgent", bb, [
            TOPICS["MERGE_RES"],
            TOPICS["DESIGN_RES"],
            TOPICS["ARBITER_RES"],
            TOPICS["VECTOR_RES"],
        ])
        self._job2entity: Dict[str, str] = {}
        self._job2style: Dict[str, str] = {}

    async def handle(self, msg: Msg):
        if msg.topic == TOPICS["MERGE_RES"]:
            merged = msg.payload.get("merged") or {}
            ek = _entity_key(merged)
            self._job2entity[msg.job_id] = ek
            # 初始化实体记忆槽
            self.bb.mem_set(f"style:{ek}", self.bb.mem_get(f"style:{ek}", {}))

        elif msg.topic == TOPICS["DESIGN_RES"]:
            style_json = msg.payload.get("style_json")
            if style_json:
                self._job2style[msg.job_id] = style_json

        elif msg.topic == TOPICS["ARBITER_RES"]:
            decision = (msg.payload or {}).get("decision")
            fused = (msg.payload or {}).get("review") or {}
            if decision != "stop":
                return
            ek = self._job2entity.get(msg.job_id, "unknown")
            style_json = self._job2style.get(msg.job_id)
            if not ek or not style_json:
                return

            now = int(time.time())
            record = self.bb.mem_get(f"style:{ek}", {})
            # 只要通过仲裁就认为是“最佳样式”的一个版本；可按分数替换/追加
            best = {
                "style_json": style_json,
                "review": fused,
                "updated_at": now,
            }
            # 你也可以在这里做“只保留分数更高的”策略
            record["best"] = best
            self.bb.mem_set(f"style:{ek}", record)

        elif msg.topic == TOPICS["VECTOR_RES"]:
            svg_path = msg.payload.get("svg_path")
            ek = self._job2entity.get(msg.job_id, "unknown")
            if not svg_path or not ek:
                return
            record = self.bb.mem_get(f"style:{ek}", {})
            record["latest_svg"] = svg_path
            self.bb.mem_set(f"style:{ek}", record)
