from ..core.agent_base import Agent
from ..core.messages import Msg, TOPICS
from ..agents.interpreter_agent import run_interpreter

class InterpreterWorker(Agent):
    def __init__(self, bb):
        super().__init__("InterpreterWorker", bb, [TOPICS["INTENT_REQ"]])

    async def handle(self, msg: Msg):
        schema = run_interpreter(msg.payload["user_text"])
        await self.bb.publish(Msg(topic=TOPICS["INTENT_RES"], job_id=msg.job_id,
                                  sender=self.name, payload={"schema": schema}))
