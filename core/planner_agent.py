# Agent/core/planner_agent.py
import asyncio
from pathlib import Path
import requests
from .agent_base import Agent
from .messages import Msg, TOPICS
import json
import os


# [新增] 简单的下载函数，带防盗链 Header
def _download_temp_image(url: str) -> str | None:
    try:
        # 存到 outputs/temp_downloads 目录下
        out_dir = Path(__file__).resolve().parents[2] / "Agent" / "outputs" / "temp_downloads"
        out_dir.mkdir(parents=True, exist_ok=True)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://image.baidu.com/"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            fname = f"auto_{hash(url)}.jpg"
            local_path = out_dir / fname
            local_path.write_bytes(resp.content)
            print(f"[Planner] ⬇️ Downloaded ref image: {local_path}", flush=True)
            return str(local_path)
    except Exception as e:
        print(f"[Planner] ⚠️ Download failed: {e}", flush=True)
    return None

class PlannerAgent(Agent):
    def __init__(self, bb, max_rounds=3):
        super().__init__("Planner", bb, [TOPICS["INTENT_REQ"], TOPICS["ARBITER_RES"]])
        self.state = {}     # job_id -> {"round":1, "style_json":..., "spec":..., ...}
        self.max_rounds = max_rounds

    async def handle(self, msg: Msg):
        if msg.topic == TOPICS["INTENT_REQ"]:
            await self._kickoff(msg)
        elif msg.topic == TOPICS["ARBITER_RES"]:
            await self._decide_next(msg)

    async def _kickoff(self, msg: Msg):
        j = msg.job_id
        user_text = msg.payload["user_text"]
        image_path = msg.payload.get("image_path")

        # =========== [新增逻辑] 智能提取实体名 ===========
        search_query = user_text  # 默认情况用原句（万一解析失败）

        print("[Planner] ⏳ Waiting for Interpreter parsing...", flush=True)
        try:
            # 等待意图解析结果（建议超时时间设长一点，防止模型响应慢）
            intent_msg = await self._await_one(j, TOPICS["INTENT_RES"], timeout=30.0, label="INTENT_RES")

            # 解析 JSON
            schema_str = intent_msg.payload.get("schema", "{}")
            schema = json.loads(schema_str)
            entity_name = schema.get("entity", {}).get("name")

            if entity_name and len(entity_name) < len(user_text):
                search_query = entity_name  # 提取成功！用短词替换长句
                print(f"[Planner] 🎯 Optimization: Use entity name '{search_query}' for searching.", flush=True)
            else:
                print("[Planner] ⚠️ Use full text for searching (no entity name found).", flush=True)

        except Exception as e:
            # 如果超时或解析失败，做一个简单的兜底截断，防止百度报错
            print(f"[Planner] ⚠️ Interpreter failed: {e}. Fallback to truncated text.", flush=True)
            if len(user_text) > 20:
                search_query = user_text[:20]  # 强制截断，避免SSL错误

        # =========== [关键修改] 分发请求 ===========

        print("[Planner] → publish GROUND_REQ / SPEC_REQ", flush=True)

        # 1. 发给 Grounder：使用"清洗后"的短词 (search_query)
        #    这样底层 agent 拿到的就是短词，不会报错，也不用改底层代码
        await self.bb.publish(Msg(topic=TOPICS["GROUND_REQ"], job_id=j, sender=self.name,
                                  payload={"user_text": search_query}))

        # 2. 发给 SpecInfer：仍然使用"原始"的长句 (user_text)
        #    因为推理 Agent 需要完整的风格描述（如"艺术化风格"、"留白"）
        await self.bb.publish(Msg(topic=TOPICS["SPEC_REQ"], job_id=j, sender=self.name,
                                  payload={"user_text": user_text}))

        detect_msg = None
        if image_path:
            print("[Planner] → publish DETECT_REQ", flush=True)
            await self.bb.publish(Msg(topic=TOPICS["DETECT_REQ"], job_id=j, sender=self.name,
                                      payload={"image_path": image_path, "schema": "{\"kind\":\"landmark\"}"}))

        # ② 等结果（Detector 可选 + 有超时）
        ground = await self._await_one(j, TOPICS["GROUND_RES"], label="GROUND_RES")
        spec = await self._await_one(j, TOPICS["SPEC_RES"], label="SPEC_RES")

        # =========== [关键修改 START] ===========
        # 4. 检查是否需要“自主视觉增强”
        # 如果一开始没图，但 Grounder 找到了图，就下载并补充检测
        detect = None

        # 尝试获取 Grounder 返回的 URL
        grounded_data = ground.payload.get("grounded", {})
        ref_url = grounded_data.get("reference_image_url")

        if not image_path and ref_url:
            print(f"[Planner] 🤖 Found URL from Grounder, downloading...", flush=True)
            downloaded_path = _download_temp_image(ref_url)

            if downloaded_path:
                print("[Planner] → publish DETECT_REQ (Auto-Visual)", flush=True)
                # 动态发起视觉检测
                await self.bb.publish(Msg(topic=TOPICS["DETECT_REQ"], job_id=j, sender=self.name,
                                          payload={"image_path": downloaded_path, "schema": "{\"kind\":\"landmark\"}"}))

                # 等待检测结果
                detect = await self._await_optional(j, TOPICS["DETECT_RES"], timeout=45.0, label="DETECT_RES(Auto)")

        # 如果一开始就有图，这里接收初始检测的结果
        elif image_path:
            detect = await self._await_optional(j, TOPICS["DETECT_RES"], timeout=10.0, label="DETECT_RES(Initial)")
        # =========== [关键修改 END] ===========

        # ③ 合并规范
        print("[Planner] → publish MERGE_REQ", flush=True)
        await self.bb.publish(Msg(topic=TOPICS["MERGE_REQ"], job_id=j, sender=self.name,
                                  payload={"user_spec": spec.payload.get("spec"),
                                           "detector_spec": (detect.payload.get("detector") if detect else {}),
                                           "defaults": ground.payload.get("grounded")}))

        merged = await self._await_one(j, TOPICS["MERGE_RES"], label="MERGE_RES")
        self.state[j] = {"round": 1, "spec": merged.payload["merged"]}

        # ④ 设计样式
        print("[Planner] → publish DESIGN_REQ", flush=True)
        await self.bb.publish(Msg(topic=TOPICS["DESIGN_REQ"], job_id=j, sender=self.name,
                                  payload={"detector_spec": (detect.payload.get("detector") if detect else "{}"),
                                           "schema": "{}", "structure_spec": merged.payload["merged"]}))
        style = await self._await_one(j, TOPICS["DESIGN_RES"], label="DESIGN_RES")
        self.state[j]["style_json"] = style.payload["style_json"]

        # ⑤ 生成候选
        print("[Planner] → publish GEN_REQ", flush=True)
        await self.bb.publish(Msg(topic=TOPICS["GEN_REQ"], job_id=j, sender=self.name,
                                  payload={"style_json": style.payload["style_json"],
                                           "user_text": user_text,
                                           "structure_spec": merged.payload["merged"]}))
        gen = await self._await_one(j, TOPICS["GEN_RES"], label="GEN_RES")
        best_png = gen.payload["best_png"]
        self.state[j]["best_png"] = best_png

        # ⑥ 并行两位审稿人
        print("[Planner] → REVIEW_STRUCT_REQ & REVIEW_AESTH_REQ")
        await self.bb.publish(Msg(topic=TOPICS["REVIEW_STRUCT_REQ"], job_id=j, sender=self.name,
                                  payload={"image_path": best_png, "structure_spec": merged.payload["merged"]}))
        await self.bb.publish(Msg(topic=TOPICS["REVIEW_AESTH_REQ"], job_id=j, sender=self.name,
                                  payload={"image_path": best_png, "structure_spec": merged.payload["merged"]}))

    async def _decide_next(self, msg: Msg):
        j = msg.job_id
        st = self.state.get(j, {"round": 1})
        decision = (msg.payload or {}).get("decision")
        fused = (msg.payload or {}).get("review") or {}

        # ====== 收敛：触发矢量化，再 DONE ======
        if decision == "stop" or st["round"] >= self.max_rounds:
            best_png = st.get("best_png")  # 在 _kickoff() / 生成阶段要把 best_png 存入 state
            svg_path = None
            if best_png:
                print("[Planner] → VECTOR_REQ", flush=True)
                await self.bb.publish(Msg(
                    topic=TOPICS["VECTOR_REQ"], job_id=j, sender=self.name,
                    payload={"png_path": best_png, "method": "auto", "simplify_eps": 1.0}
                ))
                vec = await self._await_one(j, TOPICS["VECTOR_RES"], label="VECTOR_RES")
                svg_path = (vec.payload or {}).get("svg_path")

            await self.bb.publish(Msg(
                topic=TOPICS["DONE"], job_id=j, sender=self.name,
                payload={"review": fused, "svg_path": svg_path}
            ))
            self.state.pop(j, None)
            return

        # ====== 继续细化：Designer → Generator → 双审稿人 ======
        st["round"] += 1
        await self.bb.publish(Msg(
            topic=TOPICS["REFINE_REQ"], job_id=j, sender=self.name,
            payload={
                "prev_style_json": st["style_json"],
                "review_json": fused,
                "structure_spec": st["spec"]
            }
        ))
        style = await self._await_one(j, TOPICS["DESIGN_RES"], label="DESIGN_RES")
        st["style_json"] = style.payload["style_json"]

        # 新一轮生成
        await self.bb.publish(Msg(
            topic=TOPICS["GEN_REQ"], job_id=j, sender=self.name,
            payload={
                "style_json": st["style_json"],
                "user_text": "reuse",
                "structure_spec": st["spec"]
            }
        ))
        gen = await self._await_one(j, TOPICS["GEN_RES"], label="GEN_RES")
        best_png = gen.payload["best_png"]
        st["best_png"] = best_png  # ← 记住给收敛时矢量化用

        # 并行两位审稿人（分开队列，避免互相吞消息）
        print("[Planner] → REVIEW_STRUCT_REQ & REVIEW_AESTH_REQ", flush=True)
        await self.bb.publish(Msg(
            topic=TOPICS["REVIEW_STRUCT_REQ"], job_id=j, sender=self.name,
            payload={"image_path": best_png, "structure_spec": st["spec"]}
        ))
        await self.bb.publish(Msg(
            topic=TOPICS["REVIEW_AESTH_REQ"], job_id=j, sender=self.name,
            payload={"image_path": best_png, "structure_spec": st["spec"]}
        ))

    async def _await_one(self, job_id: str, topic: str, timeout: float = 30.0, label: str = ""):
        import time, asyncio
        start = time.time()
        q = self.bb.topic(topic)
        while time.time() - start < timeout:
            msg = await asyncio.wait_for(q.get(), timeout=timeout)
            if msg.job_id == job_id:
                if label: print(f"[Planner] ✓ {label}", flush=True)
                return msg
            # 不是我这单的消息：丢回队尾，避免“吃掉别人”的消息
            await q.put(msg)
        raise TimeoutError(f"wait {topic} for job {job_id} timeout")

    async def _await_optional(self, job_id: str, topic: str, timeout: float = 2.0, label: str = ""):
        try:
            return await self._await_one(job_id, topic, timeout=timeout, label=label)
        except Exception:
            print(f"[Planner] • {label} not available, continue", flush=True)
            return None
