from __future__ import annotations

import json
import logging
from typing import Any

from src.util.redis_client import get_redis

log = logging.getLogger("sandworm")

ACTIVE_JOB_TTL = 60 * 5
JOB_EVENT_TTL = 60 * 60
JOB_KEY_PREFIX = "ai:job"


async def get_active_job(chat_id: str) -> str | None:
    return await get_redis().get(f"active_job:{chat_id}")


async def set_active_job(chat_id: str, job_id: str) -> None:
    await get_redis().set(f"active_job:{chat_id}", job_id, ex=ACTIVE_JOB_TTL)


async def clear_active_job(chat_id: str, job_id: str) -> None:
    """Only the job that is still the chat's current one clears it; a newer message has already replaced an older job."""
    if await get_redis().get(f"active_job:{chat_id}") == job_id:
        await get_redis().delete(f"active_job:{chat_id}")


# Same key ChatService.abort() sets (apps/api/.../chat.service.ts): the one
# thing Node and this service agree on across the language boundary. It is
# keyed by job, one job per user message, so stopping one message can never
# cancel the next. Polled at checkpoints throughout a chat run rather than
# pushed, since there's no way to reach into an already-running asyncio
# background task from outside.
CANCEL_JOB_TTL = 60 * 5


def _cancel_job_key(job_id: str) -> str:
    return f"cancel_job:{job_id}"


async def request_job_cancel(job_id: str) -> None:
    await get_redis().set(_cancel_job_key(job_id), "1", ex=CANCEL_JOB_TTL)


async def is_job_cancelled(job_id: str) -> bool:
    return await get_redis().get(_cancel_job_key(job_id)) is not None


async def clear_job_cancel(job_id: str) -> None:
    await get_redis().delete(_cancel_job_key(job_id))


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
