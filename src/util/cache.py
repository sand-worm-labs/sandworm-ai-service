from __future__ import annotations

import hashlib

import redis.asyncio as aioredis

_client: aioredis.Redis | None = None

LLM_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days


async def init_redis(url: str) -> None:
    global _client
    _client = aioredis.from_url(url, decode_responses=True)


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _client_or_raise() -> aioredis.Redis:
    if _client is None:
        raise RuntimeError("Redis not initialised — call init_redis() first")
    return _client


def make_llm_cache_key(system: str, user: str) -> str:
    digest = hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()
    return f"llm:v1:{digest}"


async def get_cached(key: str) -> str | None:
    return await _client_or_raise().get(key)


async def set_cached(key: str, value: str, ttl: int = LLM_CACHE_TTL) -> None:
    await _client_or_raise().set(key, value, ex=ttl)
