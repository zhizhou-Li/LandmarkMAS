from ..core.agent_base import Agent
from ..core.messages import Msg, TOPICS
from ..agents.vectorizer_agent import png_to_svg

class VectorizerWorker(Agent):
    def __init__(self, bb):
        super().__init__("VectorizerWorker", bb, [TOPICS["VECTOR_REQ"]])

    async def handle(self, msg: Msg):
        svg = png_to_svg(
            input_png=msg.payload["png_path"],
            out_svg=msg.payload.get("out_svg"),
            method=msg.payload.get("method","auto"),
            threshold=msg.payload.get("threshold",180),
            simplify_eps=msg.payload.get("simplify_eps",1.0))
        await self.bb.publish(Msg(topic=TOPICS["VECTOR_RES"], job_id=msg.job_id,
                                  sender=self.name, payload={"svg_path": svg}))
