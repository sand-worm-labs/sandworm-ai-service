from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage
from src.providers.openrouter import make_llm

from .models import PlanBlocksRequest, BlockPlan
from .prompts import SYSTEM_PROMPT


def _intent_summary(req: PlanBlocksRequest) -> str:
    intent = req.intent
    parts: list[str] = [f"Goal: {intent.goal}"]

    if intent.entity.addresses:
        addrs = ", ".join(f"{a.address} ({a.chain})" for a in intent.entity.addresses)
        parts.append(f"Addresses: {addrs}")
    if intent.entity.protocol_names:
        parts.append(f"Protocols: {', '.join(intent.entity.protocol_names)}")
    if intent.params:
        parts.append(f"Params: {json.dumps(intent.params)}")

    feasible = [sg for sg in intent.sub_goals if sg.feasible]
    if feasible:
        parts.append("Sub-goals:")
        for sg in feasible:
            parts.append(f"  - {sg.goal}")

    return "\n".join(parts)


class PlanBlocksService:
    def __init__(self, req: PlanBlocksRequest):
        self.llm = make_llm(req.openrouter_api_key, req.model)
        self.req = req

    async def plan(self) -> BlockPlan:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_intent_summary(self.req)),
        ]
        response = await self.llm.ainvoke(messages)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.content.strip())
        return BlockPlan.model_validate(json.loads(raw))
