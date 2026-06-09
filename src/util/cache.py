from __future__ import annotations

import hashlib
import json
from typing import Any

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


ACTIVE_JOB_TTL = 60 * 5   # 5 minutes — safety expiry if pipeline crashes
JOB_EVENT_TTL  = 60 * 60  # 1 hour


async def get_active_job(chat_id: str) -> str | None:
    return await _client_or_raise().get(f"active_job:{chat_id}")


async def set_active_job(chat_id: str, job_id: str) -> None:
    await _client_or_raise().set(f"active_job:{chat_id}", job_id, ex=ACTIVE_JOB_TTL)


async def clear_active_job(chat_id: str) -> None:
    await _client_or_raise().delete(f"active_job:{chat_id}")


async def publish_job_event(job_id: str, event: dict[str, Any]) -> None:
    client = _client_or_raise()
    payload = json.dumps(event)
    key = f"job:{job_id}:event"
    await client.set(key, payload, ex=JOB_EVENT_TTL)
    await client.publish(f"job:{job_id}", payload)
