from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage
from src.providers.openrouter import make_llm

from .models import PlanBlocksRequest, BlockPlan


SYSTEM_PROMPT = """You are a notebook block planner for Sandworm, a blockchain analytics platform.

Given a resolved analytics intent with sub-goals, produce an ordered sequence of notebook blocks that will fulfill the analysis. Each block maps to one of four types:

- sql        — a DuckDB/SQL query that fetches or aggregates on-chain data
- python     — data transformation, computation, or post-processing using pandas/numpy
- visualization — a plotly chart rendered from a prior SQL or Python block's output
- markdown   — a section header, insight callout, or explanatory commentary

RULES:
1. Every sub_goal marked feasible:true needs at least one sql block.
2. A visualization block must always follow a sql or python block it depends on — set depends_on to that block's 0-based index.
3. A python block is only needed when the SQL result requires non-trivial transformation (e.g. join across results, rolling window, custom metric).
4. Open with a markdown block that titles the analysis only when the plan has 3+ other blocks.
5. Each sql/python block may be followed by at most one visualization block.
6. Sub-goals marked feasible:false must be skipped entirely — do not create blocks for them.
7. Keep titles concise (≤8 words). Descriptions should say what the block does, not how.
8. depends_on lists the 0-based indices of blocks whose output this block needs.

Output ONLY valid JSON matching this schema — no markdown, no explanation:
{"blocks":[{"type":"sql|python|visualization|markdown","title":"...","description":"...","depends_on":[]},...]}"""


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
