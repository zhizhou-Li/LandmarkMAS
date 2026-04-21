# -*- coding: utf-8 -*-
from __future__ import annotations
from statistics import mean
from typing import Dict, Any, List

from .agent_base import Agent
from .messages import Msg, TOPICS
try:
    # 直接复用你项目里的阈值
    from config import TARGETS
except Exception:
    TARGETS = {"clarity": 80, "aesthetic": 80, "recognizability": 80}

class ArbiterAgent(Agent):
    """
    仲裁器：汇总多位审稿人（structure/aesthetic）的评分，做“收敛/细化”决策。
    订阅: REVIEW_RES
    产出: ARBITER_RES {decision: "stop"|"refine", review: fused_json, raw: [各审稿结果]}
    """
    def __init__(self, bb, required_kinds: List[str] = None):
        super().__init__("Arbiter", bb, [TOPICS["REVIEW_RES"]])
        self.required_kinds = required_kinds or ["structure", "aesthetic"]
        # 缓冲：job_id -> {"structure": {...}, "aesthetic": {...}}
        self.buf: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _pass_threshold(self, r: Dict[str, Any]) -> bool:
        return (
            r.get("clarity_score", 0) >= TARGETS["clarity"]
            and r.get("aesthetic_score", 0) >= TARGETS["aesthetic"]
            and r.get("recognizability_score", 0) >= TARGETS["recognizability"]
            and r.get("structure_penalty", 0) <= 20
        )

    def _fuse(self, parts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """简单但稳健的融合：结构为主，美学加权"""
        struct = parts.get("structure", {})
        aesth  = parts.get("aesthetic", {})

        # 缺谁就用另一个兜底
        clarity = mean([
            x for x in [
                struct.get("clarity_score"),
                aesth.get("clarity_score"),
            ] if isinstance(x, (int, float))
        ]) if parts else 0

        fused = {
            "clarity_score": clarity,
            "aesthetic_score": aesth.get("aesthetic_score", clarity),
            "recognizability_score": min(
                struct.get("recognizability_score", 100),
                aesth.get("recognizability_score", 100)
            ),
            "structure_penalty": struct.get("structure_penalty", 0),
            "violations": list({*struct.get("violations", []), *aesth.get("violations", [])}),
            "suggestions": (struct.get("suggestions", []) + aesth.get("suggestions", []))[:50],
        }
        return fused

    async def handle(self, msg: Msg):
        # 只处理 reviewer.result
        payload = msg.payload or {}
        kind = payload.get("kind")
        res  = payload.get("result") or {}

        if kind not in self.required_kinds:
            return

        j = msg.job_id
        slot = self.buf.setdefault(j, {})
        slot[kind] = res

        # 收齐所需 reviewer 才决策
        if not all(k in slot for k in self.required_kinds):
            return

        fused = self._fuse(slot)
        decision = "stop" if self._pass_threshold(fused) else "refine"

        await self.bb.publish(Msg(
            topic=TOPICS["ARBITER_RES"],
            job_id=j,
            sender=self.name,
            payload={
                "decision": decision,
                "review": fused,
                "raw": slot
            }
        ))
        # 用完清空，避免内存涨
        self.buf.pop(j, None)
