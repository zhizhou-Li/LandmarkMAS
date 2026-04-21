from ..core.messages import Msg, TOPICS
from ..core.agent_base import Agent
from ..agents.reviewer_agent import run_reviewer

class StructureReviewer(Agent):
    def __init__(self, bb):
        super().__init__("StructureReviewer", bb, [TOPICS["REVIEW_STRUCT_REQ"]])

    async def handle(self, msg: Msg):
        r = run_reviewer(msg.payload["image_path"], msg.payload.get("structure_spec"))
        await self.bb.publish(Msg(topic=TOPICS["REVIEW_RES"], job_id=msg.job_id,
                                  sender=self.name, payload={"kind":"structure","result": r}))

class AestheticReviewer(Agent):
    def __init__(self, bb):
        super().__init__("AestheticReviewer", bb, [TOPICS["REVIEW_AESTH_REQ"]])

    async def handle(self, msg: Msg):
        r = run_reviewer(msg.payload["image_path"], msg.payload.get("structure_spec"))
        await self.bb.publish(Msg(topic=TOPICS["REVIEW_RES"], job_id=msg.job_id,
                                  sender=self.name, payload={"kind":"aesthetic","result": r}))
