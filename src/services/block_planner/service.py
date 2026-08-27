from __future__ import annotations

import json
import logging

from pydantic import ValidationError
from langchain_core.messages import SystemMessage, HumanMessage
from src.providers.openrouter import make_llm
from src.util.llm_json import LLMJSONError, parse_json_object

from .models import PlanBlocksRequest, BlockPlan
from .prompts import SYSTEM_PROMPT

log = logging.getLogger("sandworm.block_planner")


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


MAX_PLAN_ATTEMPTS = 3


def _parse_plan(raw_content: str) -> BlockPlan:
    # Pre-parse (strip fences/prose, confirm it's a JSON object) before
    # spending a pydantic validation pass on it.
    data = parse_json_object(raw_content)
    return BlockPlan.model_validate(data)


class PlanBlocksService:
    def __init__(self, req: PlanBlocksRequest):
        self.llm = make_llm(req.openrouter_api_key, req.model)
        self.req = req

    async def plan(self) -> BlockPlan:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_intent_summary(self.req)),
        ]

        last_exc: Exception | None = None
        response = None

        for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
            response = await self.llm.ainvoke(messages)
            try:
                return _parse_plan(response.content)
            except (LLMJSONError, ValidationError) as exc:
                last_exc = exc
                log.warning(
                    "block plan not valid JSON on attempt %d/%d (%s); raw=%r",
                    attempt, MAX_PLAN_ATTEMPTS, exc, response.content[:2000],
                )
                if attempt == MAX_PLAN_ATTEMPTS:
                    break

                # Feed the model its own bad output back with a corrective
                # instruction — LLMs usually self-correct a formatting slip
                # when told exactly what was wrong — then loop and try again.
                messages = [
                    *messages,
                    response,
                    HumanMessage(
                        content=(
                            "That response was not valid JSON matching the schema. "
                            "Return ONLY the JSON object — no markdown fences, no explanation."
                        )
                    ),
                ]

        # Stop instead of failing the whole chat turn: no blocks get planned,
        # but node_complete still runs and the user gets a real (text-only)
        # response rather than a raw JSON error in the chat.
        log.error(
            "block plan not valid JSON after %d attempts (%s); falling back to "
            "an empty plan. raw=%r",
            MAX_PLAN_ATTEMPTS, last_exc, response.content[:2000] if response else None,
        )
        return BlockPlan(blocks=[])
