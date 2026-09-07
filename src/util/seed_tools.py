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
    if await collection_has_data(COLLECTION):
        log.info("collection already seeded, skipping")
        return

    tools = await _fetch_tools()
    if not tools:
        log.warning("no tools fetched from %s", CATALOG_TARBALL_URL)
        return

    log.info("seeding %d tools from %s", len(tools), CATALOG_TARBALL_URL)
    service = SandwormToolsService()
    await service.upsert(tools)
    log.info("seeding complete")
