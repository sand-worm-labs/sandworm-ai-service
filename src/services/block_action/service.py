from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage
from src.providers.openrouter import make_llm
from src.services.sandworm_tools.service import SandwormToolsService
from src.util.stream_events import StreamEnvelope

from src.services.block_planner.models import BlockPlan, PlannedBlock
from src.services.intent.models import Intent
from .model import GeneratedBlock
from .prompts import SYSTEM_PROMPTS

# These block types are inserted by the Node side as empty widgets/config —
# it doesn't apply generated freeform text to them (see addBlocks in
# apps/api/.../ai-blocks.ts), so skip the LLM call rather than generate
# content that would just be discarded.
NO_CONTENT_TYPES = {
    "pivot_table",
    "rich_text",
}


def _user_message(
    block: PlannedBlock,
    intent: Intent,
    prior_blocks: list[GeneratedBlock],
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
        if idx < len(prior_blocks):
            dep = prior_blocks[idx]
            parts.append(
                f"**Preceding block ({dep.title}):**\n```\n{dep.content[:1500]}\n```"
            )

    return "\n\n".join(parts)


class BlockActionService:
    def __init__(self, api_key: str, model: str, envelope: StreamEnvelope | None = None):
        self.llm = make_llm(api_key, model)
        self.envelope = envelope
        self.tools = SandwormToolsService()

    async def generate_blocks(self, plan: BlockPlan, intent: Intent) -> list[GeneratedBlock]:
        generated: list[GeneratedBlock] = []

        for block in plan.blocks:
            generated_block = GeneratedBlock(
                type=block.type,
                title=block.title,
                description=block.description,
                content="",
                depends_on=block.depends_on,
            )

            if self.envelope:
                await self.envelope.block_generating(generated_block.id, block.type, block.title)

            if block.type == "power_toolbox":
                content = await self._select_power_tool(block, intent, generated)
            elif block.type in NO_CONTENT_TYPES:
                content = ""
            else:
                system = SYSTEM_PROMPTS[block.type]
                user = _user_message(block, intent, generated)

                response = await self.llm.ainvoke([
                    SystemMessage(content=system),
                    HumanMessage(content=user),
                ])
                content = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", response.content.strip())

            generated_block.content = content
            generated.append(generated_block)

            if self.envelope:
                await self.envelope.block_ready(generated_block.id, block.type, block.title, content)

        return generated

    # power_toolbox has no freeform "content" — it needs a real tool_id plus
    # inputs matching that tool's declared schema. Find the best-matching
    # tool via the same embedding search used for tool selection elsewhere,
    # then have the LLM fill in its inputs from the task/intent.
    async def _select_power_tool(
        self,
        block: PlannedBlock,
        intent: Intent,
        prior_blocks: list[GeneratedBlock],
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
        user = _user_message(block, intent, prior_blocks) + f"\n\n**Tool inputs schema:**\n{schema_desc}"

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
