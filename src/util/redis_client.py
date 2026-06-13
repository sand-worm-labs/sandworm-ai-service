from __future__ import annotations

import redis.asyncio as aioredis

_client: aioredis.Redis | None = None


async def init_redis(url: str) -> None:
    global _client
    _client = aioredis.from_url(url, decode_responses=True)


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def get_redis() -> aioredis.Redis:
    if _client is None:
        raise RuntimeError("Redis not initialised — call init_redis() first")
    return _client
