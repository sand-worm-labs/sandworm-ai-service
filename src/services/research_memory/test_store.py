from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.research_memory import store
from src.services.research_memory.models import Attempt


@pytest.fixture
def qdrant(mocker):
    client = MagicMock()
    client.scroll = AsyncMock(return_value=([], None))
    client.delete = AsyncMock()
    client.upsert = AsyncMock()
    client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    mocker.patch.object(store, "get_qdrant", return_value=client)
    return client


@pytest.mark.asyncio
async def test_indexing_replaces_the_notebook_and_tags_every_chunk_with_its_workspace(qdrant):
    changed = await store.index_notebook("w1", "d1", "TVL study", "Uniswap TVL fell 12%.")

    assert changed is True
    qdrant.delete.assert_awaited_once()
    points = qdrant.upsert.await_args.kwargs["points"]
    assert [p.payload["workspace_id"] for p in points] == ["w1"]
    assert points[0].payload["kind"] == "notebook"
    assert points[0].payload["document_id"] == "d1"
    assert "Uniswap TVL fell 12%." in points[0].payload["text"]


@pytest.mark.asyncio
async def test_an_unchanged_notebook_is_not_indexed_again(qdrant):
    markdown = "same content"
    qdrant.scroll.return_value = (
        [SimpleNamespace(payload={"content_hash": hashlib.sha256(markdown.encode()).hexdigest()})],
        None,
    )

    assert await store.index_notebook("w1", "d1", "t", markdown) is False
    qdrant.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_empty_notebook_is_skipped(qdrant):
    assert await store.index_notebook("w1", "d1", "t", "   \n") is False
    qdrant.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_attempts_are_stored_with_their_outcome_failures_included(qdrant):
    attempts = [
        Attempt("add_cell", "type=sql dataSource=dune", True, "ok"),
        Attempt("run_notebook", "", False, "column 'tvl' does not exist"),
    ]

    stored = await store.index_attempts("w1", "d1", "TVL study", "run-1", attempts)

    assert stored == 2
    points = qdrant.upsert.await_args.kwargs["points"]
    assert [(p.payload["kind"], p.payload["worked"]) for p in points] == [("attempt", True), ("attempt", False)]
    assert points[1].payload["text"].startswith("FAILED: run_notebook")
    assert "column 'tvl' does not exist" in points[1].payload["text"]
    assert len({p.id for p in points}) == 2


@pytest.mark.asyncio
async def test_no_attempts_means_nothing_is_stored(qdrant):
    assert await store.index_attempts("w1", "d1", "t", "run", []) == 0
    qdrant.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_is_scoped_to_the_workspace_and_searches_by_the_questions_words(qdrant):
    await store.recall("w1", "uniswap tvl", exclude_document_id="d-now")

    flt = qdrant.query_points.await_args.kwargs["query_filter"]
    assert [(c.key, c.match.value) for c in flt.must] == [("workspace_id", "w1")]
    assert qdrant.query_points.await_args.kwargs["using"] == "text"
    assert qdrant.query_points.await_args.kwargs["query"].indices  # the words of the question, hashed


def hit(doc, score, text, kind="notebook", worked=None):
    payload = {"kind": kind, "document_id": doc, "title": f"T-{doc}", "text": text}
    if worked is not None:
        payload["worked"] = worked
    return SimpleNamespace(score=score, payload=payload)


@pytest.mark.asyncio
async def test_recall_keeps_the_best_chunk_per_notebook_and_skips_the_current_one(qdrant):
    qdrant.query_points.return_value = SimpleNamespace(
        points=[
            hit("a", 0.9, "best of a"),
            hit("a", 0.8, "second of a"),
            hit("now", 0.85, "the notebook being edited"),
            hit("b", 0.1, "too weak"),
            hit("c", 0.5, "c"),
        ]
    )

    hits = await store.recall("w1", "q", exclude_document_id="now")

    assert [(h.document_id, h.text) for h in hits] == [("a", "best of a"), ("c", "c")]


@pytest.mark.asyncio
async def test_recall_keeps_attempts_even_from_the_current_notebook(qdrant):
    qdrant.query_points.return_value = SimpleNamespace(
        points=[hit("now", 0.7, "FAILED: add_cell dune", kind="attempt", worked=False)]
    )

    hits = await store.recall("w1", "q", exclude_document_id="now")

    assert [(h.kind, h.worked) for h in hits] == [("attempt", False)]


@pytest.mark.asyncio
async def test_stored_text_is_a_sparse_word_vector_under_the_collections_vector_name(qdrant):
    await store.index_notebook("w1", "d1", "TVL study", "Uniswap TVL fell 12% on arbitrum.")

    vector = qdrant.upsert.await_args.kwargs["points"][0].vector["text"]
    assert len(vector.indices) == len(vector.values) > 3
    assert all(v > 0 for v in vector.values)


@pytest.mark.asyncio
async def test_a_question_of_only_filler_words_searches_nothing(qdrant):
    assert await store.recall("w1", "what is the", ) == []
    qdrant.query_points.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_weak_match_is_dropped_whatever_the_scale_of_the_scores(qdrant):
    def point(doc, score):
        return SimpleNamespace(score=score, payload={"kind": "notebook", "document_id": doc, "title": doc, "text": doc})

    qdrant.query_points.return_value = SimpleNamespace(points=[point("strong", 40.0), point("ok", 20.0), point("weak", 4.0)])

    hits = await store.recall("w1", "uniswap tvl")

    assert [h.document_id for h in hits] == ["strong", "ok"]
