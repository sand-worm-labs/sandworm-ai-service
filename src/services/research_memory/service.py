from __future__ import annotations

import asyncio
import logging

from src.services.research_memory.models import ATTEMPT, NOTEBOOK, Attempt, MemoryHit
from src.services.research_memory.store import index_attempts, recall

log = logging.getLogger("sandworm.research_memory")


async def recall_text(workspace_id: str, query: str, exclude_document_id: str | None = None) -> str:
    """Recalled memory as prompt text. Memory is a bonus: any failure just means no extra context."""
    if not query.strip():
        return ""
    try:
        return format_prior_research(await recall(workspace_id, query[:1000], exclude_document_id))
    except Exception:
        log.warning("could not recall memory", exc_info=True)
        return ""


_background: set[asyncio.Task] = set()


def remember_in_background(workspace_id: str, document_id: str, run_id: str, attempt: Attempt) -> None:
    """Save one attempt without making the caller wait for the embedding."""

    async def save() -> None:
        try:
            await index_attempts(workspace_id, document_id, "Untitled notebook", run_id, [attempt])
        except Exception:
            log.warning("could not remember attempt", exc_info=True)

    task = asyncio.create_task(save())
    _background.add(task)  # a task nobody references can be garbage collected mid-run
    task.add_done_callback(_background.discard)


def _snippet(text: str, size: int) -> str:
    return " ".join(text.split())[:size]


def format_prior_research(hits: list[MemoryHit]) -> str:
    notebooks = [h for h in hits if h.kind == NOTEBOOK]
    attempts = [h for h in hits if h.kind == ATTEMPT]
    parts: list[str] = []
    if notebooks:
        parts.append(
            "Prior research in this workspace that may bear on this request. Build on it: reuse what was already "
            "established, do not redo it, and name the notebook you draw on. It can be out of date, so check anything "
            "important against fresh data, and read the notebook with get_notebook when you need more of it.\n"
            + "\n".join(f'- "{h.title}" (notebook {h.document_id}): {_snippet(h.text, 500)}' for h in notebooks)
        )
    if attempts:
        parts.append(
            "Earlier attempts in this workspace, and how they went. Repeat what worked. Do not repeat what failed "
            "unless the cause is fixed: change the approach.\n"
            + "\n".join(f'- "{h.title}": {_snippet(h.text, 400)}' for h in attempts)
        )
    return "\n\n".join(parts)
