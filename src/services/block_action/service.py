from __future__ import annotations

import re

from langchain_core.messages import SystemMessage, HumanMessage
from src.providers.openrouter import make_llm
from src.util.cache import get_cached, make_llm_cache_key, set_cached

from src.services.block_planner.models import BlockPlan, PlannedBlock
from src.services.intent.models import Intent
from .model import GeneratedBlock


_SYSTEM: dict[str, str] = {
    "sql": (
        "You are a blockchain SQL query generator for Sandworm. "
        "Write a single DuckDB SQL query that fulfills the given task. "
        "Return ONLY the SQL — no explanation, no markdown fences, no preamble."
    ),
    "python": (
        "You are a blockchain Python code generator for Sandworm. "
        "Write Python code using pandas as pd. "
        "Return ONLY the code — no explanation, no markdown fences, no preamble."
    ),
    "visualization": (
        "You are a blockchain visualization code generator for Sandworm. "
        "Write plotly Python code. Assign the final figure to a variable named `fig`. "
        "Do not call fig.show(). "
        "Return ONLY the code — no explanation, no markdown fences, no preamble."
    ),
    "markdown": (
        "You are writing a markdown block for an onchain analytics notebook. "
        "Return ONLY the markdown — no explanation, no surrounding fences."
    ),
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
    def __init__(self, api_key: str, model: str):
        self.llm = make_llm(api_key, model)

    async def generate_blocks(self, plan: BlockPlan, intent: Intent) -> list[GeneratedBlock]:
        generated: list[GeneratedBlock] = []

        for block in plan.blocks:
            system = _SYSTEM[block.type]
            user = _user_message(block, intent, generated)

            response = await self.llm.ainvoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            content = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", response.content.strip())

            generated.append(GeneratedBlock(
                type=block.type,
                title=block.title,
                description=block.description,
                content=content,
                depends_on=block.depends_on,
            ))

        return generated
