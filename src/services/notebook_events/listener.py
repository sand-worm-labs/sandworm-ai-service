from __future__ import annotations

import asyncio
import json
import logging

from src.util.redis_client import get_redis
from .models import NotebookBlockResultEvent, NOTEBOOK_EVENTS_CHANNEL

log = logging.getLogger("sandworm.notebook_events")


async def _handle(raw: dict) -> None:
    if raw.get("type") != "block_result":
        return

    try:
        event = NotebookBlockResultEvent.model_validate(raw)
    except Exception as exc:
        log.warning("invalid notebook event: %s", exc)
        return

    if event.status == "error":
        log.info(
            "block errored — retry candidate: block=%s type=%s doc=%s error=%s",
            event.blockId,
            event.blockType,
            event.documentId,
            event.error,
        )
        # TODO: trigger AI retry with error context
    else:
        log.info(
            "block succeeded: block=%s type=%s doc=%s",
            event.blockId,
            event.blockType,
            event.documentId,
        )
        # TODO: trigger next step in autonomous pipeline


async def listen() -> None:
    client = get_redis()
    pubsub = client.pubsub()
    await pubsub.subscribe(NOTEBOOK_EVENTS_CHANNEL)
    log.info("subscribed to %s", NOTEBOOK_EVENTS_CHANNEL)

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                raw = json.loads(message["data"])
                await _handle(raw)
            except Exception as exc:
                log.error("notebook event error: %s", exc)
    except asyncio.CancelledError:
        log.info("notebook event listener shutting down")
    finally:
        await pubsub.unsubscribe(NOTEBOOK_EVENTS_CHANNEL)
        await pubsub.aclose()
