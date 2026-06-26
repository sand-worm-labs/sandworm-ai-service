from __future__ import annotations

import re

from langchain_core.messages import SystemMessage, HumanMessage
from src.providers.openrouter import make_llm
from src.util.cache import publish_job_event

from src.services.block_planner.models import BlockPlan, PlannedBlock
from src.services.intent.models import Intent
from .model import GeneratedBlock
from .prompts import SYSTEM_PROMPTS


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
    def __init__(self, api_key: str, model: str, job_id: str | None = None, chat_id: str | None = None):
        self.llm = make_llm(api_key, model)
        self.job_id = job_id
        self.chat_id = chat_id

    async def generate_blocks(self, plan: BlockPlan, intent: Intent) -> list[GeneratedBlock]:
        generated: list[GeneratedBlock] = []
        total = len(plan.blocks)

        for index, block in enumerate(plan.blocks):
            generated_block = GeneratedBlock(
                type=block.type,
                title=block.title,
                description=block.description,
                content="",
                depends_on=block.depends_on,
            )

            if self.job_id:
                await publish_job_event(self.job_id, {
                    "type": "generating_block",
                    "block_id": generated_block.id,
                    "block_type": block.type,
                    "block_title": block.title,
                    "index": index,
                    "total": total,
                }, self.chat_id)

            system = SYSTEM_PROMPTS[block.type]
            user = _user_message(block, intent, generated)

            response = await self.llm.ainvoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            content = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", response.content.strip())
            generated_block.content = content
            generated.append(generated_block)

            if self.job_id:
                await publish_job_event(self.job_id, {
                    "type": "block_ready",
                    "block_id": generated_block.id,
                    "block_type": block.type,
                    "block_title": block.title,
                    "content": content,
                    "index": index,
                    "total": total,
                }, self.chat_id)

        return generated
