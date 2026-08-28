from __future__ import annotations

import hashlib
import json
from typing import Any

from src.util.redis_client import get_redis

LLM_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days
ACTIVE_JOB_TTL = 60 * 5
JOB_EVENT_TTL = 60 * 60
JOB_KEY_PREFIX = "ai:job"


def make_llm_cache_key(system: str, user: str) -> str:
    digest = hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()
    return f"llm:v1:{digest}"


async def get_cached(key: str) -> str | None:
    return await get_redis().get(key)


async def set_cached(key: str, value: str, ttl: int = LLM_CACHE_TTL) -> None:
    await get_redis().set(key, value, ex=ttl)


async def get_active_job(chat_id: str) -> str | None:
    return await get_redis().get(f"active_job:{chat_id}")


async def set_active_job(chat_id: str, job_id: str) -> None:
    await get_redis().set(f"active_job:{chat_id}", job_id, ex=ACTIVE_JOB_TTL)


async def clear_active_job(chat_id: str) -> None:
    await get_redis().delete(f"active_job:{chat_id}")


async def publish_job_event(job_id: str, event: dict[str, Any], chat_id: str | None = None) -> None:
    client = get_redis()
    if chat_id is not None:
        event = {**event, "chat_id": chat_id}
    payload = json.dumps(event)
    key = f"{JOB_KEY_PREFIX}:{job_id}:events"
    async with client.pipeline() as pipe:
        await pipe.rpush(key, payload)
        await pipe.expire(key, JOB_EVENT_TTL)
        await pipe.publish(f"{JOB_KEY_PREFIX}:{job_id}", payload)
        await pipe.execute()


async def get_job_events(job_id: str) -> list[dict[str, Any]]:
    raw = await get_redis().lrange(f"{JOB_KEY_PREFIX}:{job_id}:events", 0, -1)
    return [json.loads(r) for r in raw]


# The other half of the block-result handoff — Node RPUSHes here
# (AiBlockEventService.publishBlockResult) once a block it just ran reaches a
# terminal state (success, or gives up after exhausting auto-fix retries).
# BLPOP rather than pub/sub: it still finds the value even if the push
# happened slightly before we started waiting, which pub/sub would drop.
async def wait_for_block_result(block_id: str, timeout: int = 60) -> dict[str, Any] | None:
    result = await get_redis().blpop(f"block:result:{block_id}", timeout=timeout)
    if result is None:
        return None
    _, payload = result
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
