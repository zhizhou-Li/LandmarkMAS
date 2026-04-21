from ..core.agent_base import Agent
from ..core.messages import Msg, TOPICS
from ..agents.generator_agent import run_generator
from ..agents.reviewer_agent import run_reviewer  # 新增：用它快速预评

class GeneratorWorker(Agent):
    def __init__(self, bb):
        super().__init__("GeneratorWorker", bb, [TOPICS["GEN_REQ"]])

    async def handle(self, msg: Msg):
        spec = msg.payload["structure_spec"]
        paths = run_generator(
            outline_path=msg.payload.get("outline_path"),
            style_json=msg.payload["style_json"],
            user_text=msg.payload.get("user_text",""),
            structure_spec=spec,
            base_image=msg.payload.get("base_image"),
            mask_image=msg.payload.get("mask_image"),
        )

        # —— 新增：对所有候选做一次快速结构感知打分，选最优 —— #
        scored = []
        for p in paths:
            r = run_reviewer(p, structure_spec=spec)
            score = (r.get("clarity_score",0)
                     + r.get("aesthetic_score",0)
                     + r.get("recognizability_score",0)
                     - 0.5 * r.get("structure_penalty",0))
            scored.append((p, r, score))
        best_path, best_review, _ = max(scored, key=lambda x: x[2])

        await self.bb.publish(Msg(
            topic=TOPICS["GEN_RES"], job_id=msg.job_id, sender=self.name,
            payload={
                "candidates": paths,
                "best_png": best_path,        # ← 不再是 paths[0]，而是模型评出来“最好”的
                "best_review": best_review
            }
        ))
