from __future__ import annotations

import logging

import httpx

log = logging.getLogger("sandworm.seed")

from src.config.settings import settings
from src.services.sandworm_tools.models import SandwormTool, ToolInput
from src.services.sandworm_tools.service import SandwormToolsService, COLLECTION
from src.util.qdrant import collection_has_data

GET_TOOLS_QUERY = """
query GetTools {
  getTools {
    toolId
    categoryId
    description
    tags
    params
    g1
    g2
    g3
    g4
    g5
  }
}
"""


async def _fetch_tools_from_api() -> list[SandwormTool]:
    url = f"{settings.nest_base_url.rstrip('/')}/graphql"
    async with httpx.AsyncClient() as client:
        res = await client.post(url, json={"query": GET_TOOLS_QUERY}, timeout=30)
        res.raise_for_status()
        body = res.json()

    if "errors" in body:
        raise RuntimeError(f"getTools query failed: {body['errors']}")

    tools = []
    for row in body["data"]["getTools"]:
        try:
            tags = row.get("tags") or []
            description = row["description"]
            if tags:
                description = f"{description} (tags: {', '.join(tags)})"

            inputs = [
                ToolInput(
                    key=p.get("key", ""),
                    label=p.get("label", p.get("key", "")),
                    type=p.get("type", "string"),
                    required=p.get("required", False),
                    default=p.get("default"),
                )
                for p in (row.get("params") or [])
                if isinstance(p, dict)
            ]

            tools.append(SandwormTool(
                tool_id=row["toolId"],
                g1=row.get("g1") or row.get("categoryId") or None,
                g2=row.get("g2") or None,
                g3=row.get("g3") or None,
                g4=row.get("g4") or None,
                g5=row.get("g5") or None,
                description=description,
                inputs=inputs,
            ))
        except Exception as e:
            log.warning("skipping row %s: %s", row.get("toolId"), e)
            continue
    return tools


async def seed_tools() -> None:
    if await collection_has_data(COLLECTION):
        log.info("collection already seeded, skipping")
        return

    tools = await _fetch_tools_from_api()
    if not tools:
        log.warning("no tools fetched from getTools")
        return

    log.info("seeding %d tools from getTools", len(tools))
    service = SandwormToolsService()
    await service.upsert(tools)
    log.info("seeding complete")
