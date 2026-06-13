from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

VECTOR_SIZE = 3072

COLLECTIONS: dict[str, VectorParams] = {
    "sandworm_tools": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
}

_client: AsyncQdrantClient | None = None


async def init_qdrant(url: str, api_key: str | None = None) -> None:
    global _client
    _client = AsyncQdrantClient(url=url, api_key=api_key or None)
    for name, params in COLLECTIONS.items():
        if not await _client.collection_exists(name):
            await _client.create_collection(collection_name=name, vectors_config=params)


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
