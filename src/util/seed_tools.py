from __future__ import annotations

import csv
import json
from pathlib import Path

from src.services.sandworm_tools.models import SandwormTool, ToolInput, ToolReturn
from src.services.sandworm_tools.service import SandwormToolsService, COLLECTION
from src.util.qdrant import collection_has_data


def _parse_tools(csv_path: Path) -> list[SandwormTool]:
    tools = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            try:
                raw_returns = json.loads(row.get("returns") or "[]")
                returns = [
                    ToolReturn(name=r.split(":")[0], type=r.split(":")[1] if ":" in r else "string")
                    if isinstance(r, str)
                    else ToolReturn(name=r.get("key", ""), type=r.get("type", "string"))
                    for r in raw_returns
                ]

                raw_inputs = json.loads(row.get("inputs") or "[]")
                inputs = [
                    ToolInput(
                        key=i.get("key", ""),
                        label=i.get("label", i.get("key", "")),
                        type=i.get("type", "string"),
                        required=i.get("required", False),
                        default=i.get("default"),
                    )
                    for i in raw_inputs
                    if isinstance(i, dict)
                ]

                tools.append(SandwormTool(
                    tool_id=row["tool_id"],
                    g1=row.get("g1") or None,
                    g2=row.get("g2") or None,
                    g3=row.get("g3") or None,
                    g4=row.get("g4") or None,
                    g5=row.get("g5") or None,
                    description=row.get("description", ""),
                    scope=row.get("scope", "generic"),
                    returns=returns,
                    inputs=inputs,
                ))
            except Exception as e:
                print(f"[seed_tools] skipping row {row.get('tool_id')}: {e}")
                continue
    return tools


async def seed_tools(csv_path: Path) -> None:
    service = SandwormToolsService()
    await service.ensure_collection()

    if await collection_has_data(COLLECTION):
        return

    tools = _parse_tools(csv_path)
    if not tools:
        return
    await service.upsert(tools)
