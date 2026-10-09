from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.services.research_memory import service
from src.services.research_memory.models import Attempt, MemoryHit


def test_prior_research_separates_notebooks_from_what_worked_and_failed():
    hits = [
        MemoryHit("notebook", "d1", "TVL study", "Uniswap TVL fell 12%.", 0.9),
        MemoryHit("attempt", "d2", "Fees", "FAILED: add_cell dune column missing", 0.8, False),
    ]

    text = service.format_prior_research(hits)

    assert "TVL study" in text and "d1" in text and "out of date" in text
    assert "Earlier attempts" in text and "FAILED: add_cell dune" in text
    assert "Do not repeat what failed" in text
    assert service.format_prior_research([]) == ""


def test_a_suggestion_nobody_has_run_is_neither_worked_nor_failed():
    assert Attempt("fix_sql", "request=x", None, "suggested: select 2").text().startswith("SUGGESTED: fix_sql")


@pytest.mark.asyncio
async def test_recall_text_never_raises(mocker):
    mocker.patch.object(service, "recall", new=AsyncMock(side_effect=RuntimeError("qdrant down")))

    assert await service.recall_text("w1", "q") == ""
    assert await service.recall_text("w1", "   ") == ""


@pytest.mark.asyncio
async def test_remember_in_background_saves_without_blocking(mocker):
    import asyncio

    saved = asyncio.Event()

    async def fake_index(*args):
        saved.set()

    mocker.patch.object(service, "index_attempts", new=fake_index)

    service.remember_in_background("w1", "d1", "run", Attempt("edit_sql", "x", None, "y"))
    await asyncio.wait_for(saved.wait(), timeout=1)
