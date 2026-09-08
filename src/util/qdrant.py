from __future__ import annotations

import asyncio
import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger("sandworm")

VECTOR_SIZE = 3072

COLLECTIONS: dict[str, VectorParams] = {
    "sandworm_tools": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
}

_client: AsyncQdrantClient | None = None

# Qdrant may not be reachable yet the instant this service starts (DNS/network
# not settled after a restart, container still booting, etc.), so retry with
# backoff instead of crashing the whole app on the first attempt.
_CONNECT_RETRIES = 10
_CONNECT_BACKOFF_SECONDS = 3


async def init_qdrant(url: str, api_key: str | None = None) -> None:
    global _client
    client = AsyncQdrantClient(url=url, api_key=api_key or None)

    for attempt in range(1, _CONNECT_RETRIES + 1):
        try:
            for name, params in COLLECTIONS.items():
                if not await client.collection_exists(name):
                    await client.create_collection(collection_name=name, vectors_config=params)
            break
        except Exception:
            if attempt == _CONNECT_RETRIES:
                raise
            logger.warning(
                "qdrant not reachable yet (attempt %d/%d), retrying in %ds",
                attempt,
                _CONNECT_RETRIES,
                _CONNECT_BACKOFF_SECONDS,
            )
            await asyncio.sleep(_CONNECT_BACKOFF_SECONDS)

    _client = client


async def close_qdrant() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def get_qdrant() -> AsyncQdrantClient:
    if _client is None:
        raise RuntimeError("Qdrant not initialised — call init_qdrant() first")
    return _client


async def collection_has_data(collection: str) -> bool:
    info = await get_qdrant().get_collection(collection)
    return bool(info.points_count and info.points_count > 0)
