from ..core.agent_base import Agent
from ..core.messages import Msg, TOPICS
from ..agents.spec_utils import merge_specs, normalize_spec

class MergeWorker(Agent):
    def __init__(self, bb):
        super().__init__("MergeWorker", bb, [TOPICS["MERGE_REQ"]])

    async def handle(self, msg: Msg):
        merged = normalize_spec(merge_specs(
            user_spec=msg.payload.get("user_spec"),
            detector_spec=msg.payload.get("detector_spec"),
            defaults=msg.payload.get("defaults")))
        await self.bb.publish(Msg(topic=TOPICS["MERGE_RES"], job_id=msg.job_id,
                                  sender=self.name, payload={"merged": merged}))
