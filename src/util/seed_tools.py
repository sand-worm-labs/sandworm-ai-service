from __future__ import annotations

import io
import logging
import tarfile

import httpx
import yaml

log = logging.getLogger("sandworm.seed")

from src.services.sandworm_tools.models import SandwormTool
from src.services.sandworm_tools.service import SandwormToolsService, COLLECTION
from src.util.qdrant import collection_has_data

# Fetched live over HTTP on every boot rather than vendored on disk (no git
# submodule, no local copy) — sand-worm-labs/tools is the single source of
# truth. One request for the whole catalog, parsed entirely in memory.
CATALOG_TARBALL_URL = "https://codeload.github.com/sand-worm-labs/tools/tar.gz/refs/heads/main"


async def _fetch_tools() -> list[SandwormTool]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        response = await client.get(CATALOG_TARBALL_URL)
        response.raise_for_status()

    # Only tool.yaml (metadata) matters here — this service only picks which
    # tool + what inputs via semantic search, it never renders or executes
    # anything, so the sibling template.py is irrelevant to it.
    tools: list[SandwormTool] = []
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or "/catalog/" not in member.name or not member.name.endswith("/tool.yaml"):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            try:
                data = yaml.safe_load(extracted.read())
                tools.append(SandwormTool(**data))
            except Exception as e:
                log.warning("skipping %s: %s", member.name, e)
    return tools


async def seed_tools() -> None:
    catalog = await _fetch_tools()
    if not catalog:
        log.warning("no tools fetched from %s", CATALOG_TARBALL_URL)
        return

    # Tools marked `implemented: false` are never offered: keep them out of the
    # index, and drop any indexed before they were marked (seeding below only
    # runs on an empty collection).
    tools = [t for t in catalog if t.implemented]
    unimplemented = [t.tool_id for t in catalog if not t.implemented]
    service = SandwormToolsService()
    has_data = await collection_has_data(COLLECTION)

    if unimplemented and has_data:
        await service.delete(unimplemented)
        log.info("removed %d unimplemented tools from the index", len(unimplemented))

    if has_data:
        log.info("collection already seeded, skipping")
        return

    log.info("seeding %d tools from %s (%d unimplemented skipped)", len(tools), CATALOG_TARBALL_URL, len(unimplemented))
    await service.upsert(tools)
    log.info("seeding complete")
