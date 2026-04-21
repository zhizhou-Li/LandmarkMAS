from ..core.agent_base import Agent
from ..core.messages import Msg, TOPICS
from ..agents.grounder_agent import ground_entity_to_spec

class GrounderWorker(Agent):
    def __init__(self, bb):
        super().__init__("GrounderWorker", bb, [TOPICS["GROUND_REQ"]])

    async def handle(self, msg: Msg):
        spec = ground_entity_to_spec(msg.payload["user_text"])

        await self.bb.publish(Msg(topic=TOPICS["GROUND_RES"], job_id=msg.job_id,
                                  sender=self.name, payload={"grounded": spec}))
