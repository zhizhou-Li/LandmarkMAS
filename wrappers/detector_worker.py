from ..core.agent_base import Agent
from ..core.messages import Msg, TOPICS
from ..agents.detector_agent import run_detector

class DetectorWorker(Agent):
    def __init__(self, bb):
        super().__init__("DetectorWorker", bb, [TOPICS["DETECT_REQ"]])

    async def handle(self, msg: Msg):
        det = run_detector(msg.payload["image_path"], msg.payload.get("schema","{}"))
        await self.bb.publish(Msg(topic=TOPICS["DETECT_RES"], job_id=msg.job_id,
                                  sender=self.name, payload={"detector": det}))
