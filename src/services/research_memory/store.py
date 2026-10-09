from __future__ import annotations

import hashlib
import uuid
from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
)

from src.services.research_memory.chunking import chunk_notebook
from src.services.research_memory.models import ATTEMPT, NOTEBOOK, Attempt, MemoryHit
from src.services.research_memory.text_vector import document_vector, query_vector
from src.util.qdrant import get_qdrant

# What is stored: a workspace's notebooks (queries, results, notes) and every
# attempt the AI made in them, worked or not. Uploaded files are not: only a
# notebook's own content is read, through the MCP's get_notebook.
COLLECTION = "notebook_memory_text"
VECTOR = "text"  # the sparse vector's name in the collection
# BM25 scores have no fixed scale, so what counts as related is relative: a
# memory must score at least this share of the best match to be mentioned.
MIN_SCORE_SHARE = 0.4
_NAMESPACE = uuid.UUID("5f0a3c1e-6a2b-4c53-9a47-0d6f4c7e9b11")


def _filter(workspace_id: str, **match: str) -> Filter:
    conditions = [FieldCondition(key="workspace_id", match=MatchValue(value=workspace_id))]
    conditions += [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in match.items()]
    return Filter(must=conditions)


async def index_notebook(workspace_id: str, document_id: str, title: str, markdown: str) -> bool:
    """Store a notebook's content for later recall. Returns False when nothing changed."""
    if not markdown.strip():
        return False
    client = get_qdrant()
    content_hash = hashlib.sha256(markdown.encode()).hexdigest()
    scope = _filter(workspace_id, kind=NOTEBOOK, document_id=document_id)

    existing, _ = await client.scroll(COLLECTION, scroll_filter=scope, limit=1, with_payload=True)
    if existing and existing[0].payload.get("content_hash") == content_hash:
        return False  # unchanged since last time: nothing to re-index

    chunks = chunk_notebook(title, markdown)

    await client.delete(COLLECTION, points_selector=FilterSelector(filter=scope))
    await client.upsert(
        COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, f"{workspace_id}:{document_id}:{n}")),
                vector={VECTOR: document_vector(chunk)},
                payload={
                    "kind": NOTEBOOK,
                    "workspace_id": workspace_id,
                    "document_id": document_id,
                    "title": title,
                    "chunk": n,
                    "text": chunk,
                    "content_hash": content_hash,
                },
            )
            for n, chunk in enumerate(chunks)
        ],
    )
    return True


async def index_attempts(workspace_id: str, document_id: str, title: str, run_id: str, attempts: list[Attempt]) -> int:
    """Store what the AI tried in one chat run, successes and failures alike."""
    if not attempts:
        return 0
    texts = [a.text() for a in attempts]
    await get_qdrant().upsert(
        COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, f"{workspace_id}:{document_id}:{run_id}:{n}")),
                vector={VECTOR: document_vector(text)},
                payload={
                    "kind": ATTEMPT,
                    "workspace_id": workspace_id,
                    "document_id": document_id,
                    "title": title,
                    "tool": a.tool,
                    "worked": a.worked,
                    "text": text,
                },
            )
            for n, (a, text) in enumerate(zip(attempts, texts))
        ],
    )
    return len(attempts)


async def recall(
    workspace_id: str, query: str, exclude_document_id: str | None = None, limit: int = 5
) -> list[MemoryHit]:
    """Past notebooks and attempts in this workspace that bear on the query.

    Notebooks: best match per notebook, leaving out the one being worked on.
    Attempts: kept from every notebook, the current one included, since what
    already failed here is exactly what not to repeat.
    Always scoped to one workspace: something is only recalled for people who
    can already open the notebook it came from.
    """
    vector = query_vector(query)
    if not vector.indices:
        return []  # nothing in the question but filler words
    result = await get_qdrant().query_points(
        COLLECTION,
        query=vector,
        using=VECTOR,
        query_filter=_filter(workspace_id),
        limit=limit * 6,  # several chunks can come from one notebook
        with_payload=True,
    )

    notebooks: dict[str, MemoryHit] = {}
    attempts: list[MemoryHit] = []
    best_score = max((p.score for p in result.points), default=0.0)
    for point in result.points:
        if point.score <= 0 or point.score < best_score * MIN_SCORE_SHARE:
            continue
        p: dict[str, Any] = point.payload
        if p.get("kind") == ATTEMPT:
            attempts.append(MemoryHit(ATTEMPT, p["document_id"], p["title"], p["text"], point.score, p.get("worked")))
        elif p["document_id"] != exclude_document_id and p["document_id"] not in notebooks:
            notebooks[p["document_id"]] = MemoryHit(NOTEBOOK, p["document_id"], p["title"], p["text"], point.score)
    return [*list(notebooks.values())[:limit], *attempts[:limit]]
