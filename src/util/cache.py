from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from src.util.redis_client import get_redis

log = logging.getLogger("sandworm")

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


# Same key ChatService.abort() sets (apps/api/.../chat.service.ts) and the
# Node-side auto-fix loops check (ai-block-event.service.ts) — the one thing
# all three have to agree on across the language boundary. Polled at
# checkpoints throughout run_pipeline rather than pushed, since there's no
# way to reach into an already-running asyncio background task from outside.
def _cancel_job_key(chat_id: str) -> str:
    return f"cancel_job:{chat_id}"


async def is_job_cancelled(chat_id: str) -> bool:
    return await get_redis().get(_cancel_job_key(chat_id)) is not None


async def clear_job_cancel(chat_id: str) -> None:
    await get_redis().delete(_cancel_job_key(chat_id))


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


