from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.util.cache import publish_job_event


@dataclass
class StreamEnvelope:
    """Emits job progress as a Claude-Messages-API-style stream:
    message_start -> content_block_start/delta/stop* -> message_delta -> message_stop.
    Owns block-index bookkeeping for one job so callers just report what happened.
    """

    job_id: str
    chat_id: str | None
    _next_index: int = field(default=0, init=False)
    _text_index: int | None = field(default=None, init=False)
    _open_blocks: dict[str, int] = field(default_factory=dict, init=False)

    async def _emit(self, payload: dict[str, Any]) -> None:
        await publish_job_event(self.job_id, payload, self.chat_id)

    async def message_start(self) -> None:
        await self._emit({
            "type": "message_start",
            "message": {"id": self.job_id, "chat_id": self.chat_id},
        })

    async def thinking(self, thinking: str, duration_ms: int) -> None:
        index = self._next_index
        self._next_index += 1
        await self._emit({
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "thinking", "thinking": ""},
        })
        await self._emit({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": thinking, "duration_ms": duration_ms},
        })
        await self._emit({"type": "content_block_stop", "index": index})

    async def block_generating(self, block_id: str, block_type: str, block_title: str) -> None:
        index = self._next_index
        self._next_index += 1
        self._open_blocks[block_id] = index
        await self._emit({
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "block_action",
                "action": "generating",
                "block_id": block_id,
                "block_type": block_type,
                "block_title": block_title,
            },
        })

    async def block_ready(self, block_id: str, block_type: str, block_title: str, content: str) -> None:
        index = self._open_blocks.pop(block_id, None)
        if index is None:
            return
        await self._emit({
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "block_action_delta",
                "action": "ran",
                "block_id": block_id,
                "block_type": block_type,
                "block_title": block_title,
                "content": content,
            },
        })
        await self._emit({"type": "content_block_stop", "index": index})

    async def text_delta(self, token: str) -> None:
        if self._text_index is None:
            self._text_index = self._next_index
            self._next_index += 1
            await self._emit({
                "type": "content_block_start",
                "index": self._text_index,
                "content_block": {"type": "text", "text": ""},
            })
        await self._emit({
            "type": "content_block_delta",
            "index": self._text_index,
            "delta": {"type": "text_delta", "text": token},
        })

    async def follow_up(self, message: str, questions: list[dict]) -> None:
        await self._emit({
            "type": "message_delta",
            "delta": {"follow_up": {"message": message, "questions": questions}},
        })

    async def message_stop(self) -> None:
        if self._text_index is not None:
            await self._emit({"type": "content_block_stop", "index": self._text_index})
        await self._emit({"type": "message_delta", "delta": {"stop_reason": "end_turn"}})
        await self._emit({"type": "message_stop"})

    async def error(self, error_type: str, message: str) -> None:
        await self._emit({"type": "error", "error": {"type": error_type, "message": message}})
