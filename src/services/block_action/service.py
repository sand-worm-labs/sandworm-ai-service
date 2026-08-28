from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage
from src.providers.openrouter import make_llm
from src.services.sandworm_tools.service import SandwormToolsService
from src.util.cache import wait_for_block_result
from src.util.stream_events import StreamEnvelope

from src.services.block_planner.models import BlockPlan, PlannedBlock
from src.services.intent.models import Intent
from .model import GeneratedBlock
from .prompts import SYSTEM_PROMPTS

# How long to wait for a dependency block's real execution result before
# giving up and falling back to generating off its code alone. Node's own
# auto-fix loop can take a few attempts, so this needs headroom beyond a
# single query's runtime.
BLOCK_RESULT_WAIT_TIMEOUT = 60

# Only these block types actually execute (and so ever get a real result
# pushed by Node's publishBlockResult) — waiting on anything else would
# just burn the full timeout for nothing.
EXECUTABLE_TYPES = {"sql", "python"}

# These block types are inserted by the Node side as empty widgets/config —
# it doesn't apply generated freeform text to them (see addBlocks in
# apps/api/.../ai-blocks.ts), so skip the LLM call rather than generate
# content that would just be discarded.
NO_CONTENT_TYPES = {
    "pivot_table",
}


def _user_message(
    block: PlannedBlock,
    intent: Intent,
    prior_blocks: list[GeneratedBlock],
    block_results: dict[int, dict | None] | None = None,
) -> str:
    parts: list[str] = [
        f"**Analytical goal:** {intent.goal}",
        f"**Task:** {block.description}",
    ]

    if intent.entity.addresses:
        addrs = ", ".join(f"{a.address} ({a.chain})" for a in intent.entity.addresses)
        parts.append(f"**Addresses:** {addrs}")
    if intent.entity.protocol_names:
        parts.append(f"**Protocols:** {', '.join(intent.entity.protocol_names)}")
    if intent.params:
        for k, v in intent.params.items():
            parts.append(f"**{k}:** {v}")

    for idx in block.depends_on:
        if idx >= len(prior_blocks):
            continue
        dep = prior_blocks[idx]
        parts.append(
            f"**Preceding block ({dep.title}):**\n```\n{dep.content[:1500]}\n```"
        )

        # If we actually waited for this dependency and it ran, tell the model
        # what really happened — not just the code we hoped would work. This
        # is the planner's real sensory input: a failed dependency should
        # steer the next block away from assuming clean data, and a successful
        # one confirms the shape/volume of what's actually available.
        outcome = (block_results or {}).get(idx)
        if outcome:
            if outcome.get("outcome") == "error":
                parts.append(
                    f"**Actual result of preceding block:** it FAILED even after "
                    f"auto-fix attempts — error: {outcome.get('summary')}. Account "
                    f"for this: do not assume the data above is available or correct; "
                    f"adapt this block accordingly (e.g. handle missing/empty data)."
                )
            else:
                parts.append(
                    f"**Actual result of preceding block:** it ran successfully — "
                    f"{outcome.get('summary')}."
                )

        if dep.type == "sql" and dep.dataframe_name:
            # The dependency already ran against Dune (or a prior local step) and
            # its result is sitting in this same session as a DataFrame — point
            # the model at it by name instead of letting it re-fetch from Dune.
            parts.append(
                f"The result of the preceding block above is already loaded in this "
                f"session as a table/DataFrame named `{dep.dataframe_name}`. Query it "
                f"directly (e.g. `SELECT ... FROM {dep.dataframe_name} ...`) — do not "
                f"re-fetch this data from Dune."
            )

    return "\n\n".join(parts)


def _dataframe_name(job_id: str | None, index: int) -> str:
    # A short, valid-identifier, per-job-unique name so a downstream block can
    # reference a preceding SQL block's result without colliding with any
    # dataframeName a human (or an earlier AI run) already has in the doc.
    token = re.sub(r"[^a-zA-Z0-9]", "", job_id or "")[:8] or "adhoc"
    return f"aiq_{token}_{index}"


def _sql_data_source(block: PlannedBlock, plan: BlockPlan) -> str:
    # If this query depends on another SQL block, that block's result is
    # already sitting in the session as a DataFrame — manipulate it locally
    # with DuckDB rather than hitting Dune again. Otherwise it's an initial
    # data pull, which only Dune (the Trino-backed source) can serve.
    for idx in block.depends_on:
        if idx < len(plan.blocks) and plan.blocks[idx].type == "sql":
            return "duckdb"
    return "dune"


class BlockActionService:
    def __init__(self, api_key: str, model: str, envelope: StreamEnvelope | None = None):
        self.llm = make_llm(api_key, model)
        self.envelope = envelope
        self.tools = SandwormToolsService()

    async def generate_blocks(self, plan: BlockPlan, intent: Intent) -> list[GeneratedBlock]:
        generated: list[GeneratedBlock] = []
        # Real execution outcomes for already-generated blocks, keyed by their
        # index in plan.blocks — populated as we wait on each one's dependents.
        block_results: dict[int, dict | None] = {}

        job_id = self.envelope.job_id if self.envelope else None

        for index, block in enumerate(plan.blocks):
            await self._await_dependency_results(block, generated, block_results)

            generated_block = GeneratedBlock(
                type=block.type,
                title=block.title,
                description=block.description,
                content="",
                depends_on=block.depends_on,
            )

            if block.type == "sql":
                generated_block.data_source = _sql_data_source(block, plan)
                generated_block.dataframe_name = _dataframe_name(job_id, index)

            if self.envelope:
                await self.envelope.block_generating(generated_block.id, block.type, block.title)

            if block.type == "power_toolbox":
                content = await self._select_power_tool(block, intent, generated, block_results)
            elif block.type in NO_CONTENT_TYPES:
                content = ""
            else:
                prompt_key = "sql_duckdb" if generated_block.data_source == "duckdb" else block.type
                system = SYSTEM_PROMPTS[prompt_key]
                user = _user_message(block, intent, generated, block_results)

                response = await self.llm.ainvoke([
                    SystemMessage(content=system),
                    HumanMessage(content=user),
                ])
                content = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", response.content.strip())

            generated_block.content = content
            generated.append(generated_block)

            if self.envelope:
                # Only sql/python blocks actually execute ("ran"). A notebook
                # has at most one dashboard_header — generating it always
                # updates that existing block in place (see
                # upsertDashboardHeaderBlock on the Node side), so it's
                # "edited", never "created". Everything else is freshly
                # inserted into the notebook, so "created".
                if block.type in EXECUTABLE_TYPES:
                    action = "ran"
                elif block.type == "dashboard_header":
                    action = "edited"
                else:
                    action = "created"
                await self.envelope.block_ready(
                    generated_block.id, block.type, block.title, content,
                    data_source=generated_block.data_source,
                    dataframe_name=generated_block.dataframe_name,
                    action=action,
                )

            # This block itself may execute (sql/python) — if a later block
            # depends on it, that block's loop iteration will wait on it via
            # _await_dependency_results above. Nothing to do here; the wait
            # is pulled by the dependent, not pushed by this block.

        return generated

    # Waits, one dependency at a time, for the real execution outcome of any
    # not-yet-resolved dependency of `block` that actually runs (sql/python).
    # This is what makes generation "interactive": Node has already inserted
    # and auto-fix-run each preceding block by the time its block_ready event
    # was processed, so by the time we get here the result may already be
    # sitting in Redis — or we block until it lands (up to the timeout), then
    # move on with whatever we got (including nothing, on timeout).
    async def _await_dependency_results(
        self,
        block: PlannedBlock,
        generated: list[GeneratedBlock],
        block_results: dict[int, dict | None],
    ) -> None:
        for dep_idx in block.depends_on:
            if dep_idx in block_results or dep_idx >= len(generated):
                continue
            dep = generated[dep_idx]
            if dep.type not in EXECUTABLE_TYPES:
                continue
            block_results[dep_idx] = await wait_for_block_result(
                dep.id, timeout=BLOCK_RESULT_WAIT_TIMEOUT
            )

    # power_toolbox has no freeform "content" — it needs a real tool_id plus
    # inputs matching that tool's declared schema. Find the best-matching
    # tool via the same embedding search used for tool selection elsewhere,
    # then have the LLM fill in its inputs from the task/intent.
    async def _select_power_tool(
        self,
        block: PlannedBlock,
        intent: Intent,
        prior_blocks: list[GeneratedBlock],
        block_results: dict[int, dict | None] | None = None,
    ) -> str:
        matches = await self.tools.search(query=block.description, top_k=1)
        if not matches:
            return json.dumps({"tool_id": None, "inputs": {}})

        tool = matches[0]
        tool_id = tool["tool_id"]
        inputs_schema = tool.get("inputs") or []

        if not inputs_schema:
            return json.dumps({"tool_id": tool_id, "inputs": {}})

        schema_desc = "\n".join(
            f"- {i['key']} ({i['type']}{', required' if i.get('required') else ''}): {i['label']}"
            for i in inputs_schema
        )
        system = (
            "You are filling in the inputs for a Sandworm power tool, based on the "
            "task. Return ONLY a JSON object mapping each input key to its value — "
            "no markdown, no explanation, no keys outside the given schema."
        )
        user = _user_message(block, intent, prior_blocks, block_results) + f"\n\n**Tool inputs schema:**\n{schema_desc}"

        response = await self.llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.content.strip())
        try:
            values = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            values = {}

        return json.dumps({"tool_id": tool_id, "inputs": values})
