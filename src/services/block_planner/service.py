from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError
from langchain_core.messages import SystemMessage, HumanMessage
from src.providers.openrouter import make_llm
from src.services.intent.models import Intent
from src.services.open_data.service import planner_context
from src.services.sandworm_tools.service import SandwormToolsService
from src.util.llm_json import LLMJSONError, parse_json_object

from .models import PlanBlocksRequest, BlockPlan, PlannedBlock
from .prompts import SYSTEM_PROMPT

log = logging.getLogger("sandworm.block_planner")

# How many candidate tools to surface per sub-goal — enough for the planner
# to recognize a real fit without bloating the prompt with irrelevant ones.
POSSIBLE_TOOLS_PER_SUBGOAL = 3


async def _search_possible_tools(tools: SandwormToolsService, intent: Intent, api_key: str) -> list[dict[str, Any]]:
    # Search per sub-goal (each already a decomposed piece of the analysis)
    # rather than one broad query on intent.goal — sharper matches for
    # multi-part asks. Falls back to the overall goal when there are no
    # feasible sub-goals at all.
    queries = [sg.goal for sg in intent.sub_goals if sg.feasible] or [intent.goal]

    seen: dict[str, dict[str, Any]] = {}
    for query in queries:
        matches = await tools.search(query=query, top_k=POSSIBLE_TOOLS_PER_SUBGOAL, api_key=api_key)
        for match in matches:
            seen.setdefault(match["tool_id"], match)

    return list(seen.values())


def _format_possible_tools(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return ""

    lines = [
        "**Possibly relevant existing tools:** only plan a power_toolbox block "
        "for a need one of these actually covers — matching by description "
        "alone isn't enough. Check each tool's required inputs too: you must "
        "be able to fill every one with a real value already pinned down by "
        "the intent (an address, a date, a number) — not a guess. If nothing "
        "here both fits AND has fillable required inputs, use a different "
        "block type (sql/python) for that sub-goal instead of guessing a "
        "power_toolbox block into existence."
    ]
    for tool in tools:
        tags = " > ".join(t for t in [tool.get("g1"), tool.get("g2"), tool.get("g3"), tool.get("g4"), tool.get("g5")] if t)
        inputs_schema = tool.get("inputs") or []
        inputs_desc = ", ".join(
            f"{i['key']}{' (required)' if i.get('required') else ''}"
            for i in inputs_schema
        ) or "none"
        lines.append(f"- {tool['tool_id']} ({tags}): {tool['description']} | inputs: {inputs_desc}")

    return "\n".join(lines)


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

# Block types that only make sense on top of another block's output.
DEPENDENT_TYPES = {"visualization", "pivot_table"}


# On open data only, the planner is told to plan python fetches only,
# but an LLM can still slip in a Dune pull or a power tool. Drop those, and
# anything left with nothing to read from, so no dead block reaches generation.
def drop_offline_blocks(plan: BlockPlan) -> BlockPlan:
    kept: list[PlannedBlock] = []
    index_map: dict[int, int] = {}

    for old_idx, block in enumerate(plan.blocks):
        depends_on = [index_map[d] for d in block.depends_on if d in index_map]
        needs_dune = block.type == "sql" and not depends_on
        orphaned = block.type in DEPENDENT_TYPES and not depends_on
        if block.type == "power_toolbox" or needs_dune or orphaned:
            log.info("dropping %s block %r: open data only", block.type, block.title)
            continue

        index_map[old_idx] = len(kept)
        kept.append(block.model_copy(update={"depends_on": depends_on}))

    return BlockPlan(blocks=kept)


def _parse_plan(raw_content: str) -> BlockPlan:
    # Pre-parse (strip fences/prose, confirm it's a JSON object) before
    # spending a pydantic validation pass on it.
    data = parse_json_object(raw_content)
    return BlockPlan.model_validate(data)


class PlanBlocksService:
    def __init__(self, req: PlanBlocksRequest):
        self.llm = make_llm(req.openrouter_api_key, req.model)
        self.req = req
        self.tools = SandwormToolsService()

    # Belt-and-suspenders: _search_possible_tools already shows the planner
    # only real candidates before it plans, but an LLM can still ignore that
    # and name a power_toolbox block anyway. Re-check with the same search
    # BlockActionService uses at generation time (_select_power_tool) so a
    # block with no real matching tool never survives into the plan, rather
    # than reaching generation and becoming a dead tool_id: None block.
    async def _drop_unmatched_power_toolbox_blocks(self, plan: BlockPlan) -> BlockPlan:
        kept: list[PlannedBlock] = []
        index_map: dict[int, int] = {}

        for old_idx, block in enumerate(plan.blocks):
            if block.type == "power_toolbox":
                matches = await self.tools.search(query=block.description, top_k=1, api_key=self.req.openrouter_api_key)
                if not matches:
                    log.info("dropping power_toolbox block %r — no matching tool found", block.title)
                    continue

            index_map[old_idx] = len(kept)
            kept.append(block)

        for block in kept:
            block.depends_on = [index_map[d] for d in block.depends_on if d in index_map]

        return BlockPlan(blocks=kept)

    async def plan(self) -> BlockPlan:
        intent = self.req.intent
        intent_summary = _intent_summary(self.req)

        if self.req.open_data:
            # No power tools on open data, so there is nothing to search.
            queries = [sg.goal for sg in intent.sub_goals if sg.feasible] or [intent.goal]
            intent_summary = f"{intent_summary}\n\n{planner_context(queries)}"
        else:
            possible_tools = await _search_possible_tools(self.tools, intent, self.req.openrouter_api_key)
            tools_context = _format_possible_tools(possible_tools)
            if tools_context:
                intent_summary = f"{intent_summary}\n\n{tools_context}"

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=intent_summary),
        ]

        last_exc: Exception | None = None
        response = None

        for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
            response = await self.llm.ainvoke(messages)
            try:
                plan = _parse_plan(response.content)
                if self.req.open_data:
                    return drop_offline_blocks(plan)
                return await self._drop_unmatched_power_toolbox_blocks(plan)
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
