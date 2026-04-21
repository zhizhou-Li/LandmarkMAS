from ..core.agent_base import Agent
from ..core.messages import Msg, TOPICS
from ..agents.spec_infer_agent import infer_structure_spec

class SpecInferWorker(Agent):
    def __init__(self, bb):
        super().__init__("SpecInferWorker", bb, [TOPICS["SPEC_REQ"]])

    async def handle(self, msg: Msg):
        spec = infer_structure_spec(
            user_text=msg.payload["user_text"],
            detector_spec=msg.payload.get("detector_spec"))
        await self.bb.publish(Msg(topic=TOPICS["SPEC_RES"], job_id=msg.job_id,
                                  sender=self.name, payload={"spec": spec}))
