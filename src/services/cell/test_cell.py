from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.base import DocumentContext
from src.services.agent.cell_edit import CellEditFailed
from src.services.cell import service as cell

CONTEXT = DocumentContext(user_id="u", workspace_id="w1", document_id="d1", user_token="tok")


@pytest.fixture
def mcp(mocker):
    client = MagicMock()
    client.aclose = AsyncMock()
    mocker.patch.object(cell, "McpClient", return_value=client)
    return client


@pytest.fixture
def write(mocker):
    return mocker.patch.object(cell, "update_cell_with_mcp", new=AsyncMock(return_value="select 2"))


@pytest.fixture
def memory(mocker):
    mocker.patch.object(cell, "recall_text", new=AsyncMock(return_value=""))
    return mocker.patch.object(cell, "remember_in_background")


@pytest.mark.parametrize("kind", ["python", "sql", "markdown"])
def test_edit_prompts_ask_for_the_whole_new_text_via_update_cell(kind):
    prompt = cell.edit_prompt(kind)

    assert "update_cell" in prompt and "COMPLETE new" in prompt and "code fences" in prompt


def test_each_cell_type_gets_its_own_persona_and_noun():
    assert "Python data scientist" in cell.edit_prompt("python")
    assert "SQL data analyst" in cell.edit_prompt("sql")
    assert "technical writer" in cell.edit_prompt("markdown")
    assert "Fix the SQL query based on the error message" in cell.fix_prompt("sql")


@pytest.mark.asyncio
async def test_edit_changes_the_cell_through_the_mcp_and_remembers_it_as_worked(mcp, write, memory):
    await cell.CellService("k", "m", "sql", CONTEXT, "c1").edit("make it 2")

    server, key, model, system, user, ref = write.await_args.args
    assert (key, model, user) == ("k", "m", "make it 2")
    assert system.startswith(cell.edit_prompt("sql"))
    assert (ref.workspace_id, ref.document_id, ref.cell_id) == ("w1", "d1", "c1")
    attempt = memory.call_args.args[3]
    assert attempt.tool == "edit_sql" and attempt.worked is True and "select 2" in attempt.result
    mcp.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_what_the_workspace_learned_goes_into_the_prompt(mcp, write, mocker):
    mocker.patch.object(cell, "recall_text", new=AsyncMock(return_value="Earlier attempts: FAILED: add_cell dune"))
    mocker.patch.object(cell, "remember_in_background")

    await cell.CellService("k", "m", "sql", CONTEXT, "c1").edit("x")

    assert "Earlier attempts: FAILED: add_cell dune" in write.await_args.args[3]


@pytest.mark.asyncio
async def test_a_failed_fix_is_remembered_as_failed_and_raised(mcp, memory, mocker):
    mocker.patch.object(cell, "update_cell_with_mcp", new=AsyncMock(side_effect=CellEditFailed("cell is running")))

    with pytest.raises(CellEditFailed):
        await cell.CellService("k", "m", "python", CONTEXT, "c1").fix("NameError")

    attempt = memory.call_args.args[3]
    assert attempt.tool == "fix_python" and attempt.worked is False and "cell is running" in attempt.result
    mcp.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_editing_needs_the_users_token(mcp, write, memory):
    no_token = DocumentContext(user_id="u", workspace_id="w1", document_id="d1")

    with pytest.raises(ValueError, match="token"):
        await cell.CellService("k", "m", "sql", no_token, "c1").edit("x")
    write.assert_not_awaited()


@pytest.mark.asyncio
async def test_markdown_cells_have_nothing_to_fix(mcp, write, memory):
    with pytest.raises(ValueError, match="nothing to fix"):
        await cell.CellService("k", "m", "markdown", CONTEXT, "c1").fix("boom")
