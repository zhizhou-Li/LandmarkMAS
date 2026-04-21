from ..core.agent_base import Agent
from ..core.messages import Msg, TOPICS
from ..agents.designer_agent import run_designer, refine_designer

class DesignerWorker(Agent):
    def __init__(self, bb):
        super().__init__("DesignerWorker", bb, [TOPICS["DESIGN_REQ"], TOPICS["REFINE_REQ"]])

    async def handle(self, msg: Msg):
        if msg.topic == TOPICS["DESIGN_REQ"]:
            sj = run_designer(
                landmark_json=msg.payload.get("detector_spec","{}"),
                schema=msg.payload.get("schema","{}"),
                structure_spec=msg.payload.get("structure_spec"))
        else:
            sj = refine_designer(
                prev_style_json=msg.payload["prev_style_json"],
                review_data=msg.payload["review_json"],
                structure_spec=msg.payload.get("structure_spec"))
        await self.bb.publish(Msg(topic=TOPICS["DESIGN_RES"], job_id=msg.job_id,
                                  sender=self.name, payload={"style_json": sj}))
