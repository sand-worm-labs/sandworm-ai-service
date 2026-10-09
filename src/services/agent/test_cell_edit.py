from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.agent import cell_edit
from src.services.agent.cell_edit import CellEditFailed, CellRef

CELL = CellRef(workspace_id="w1", document_id="d1", cell_id="c1")
SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in ("notebookId", "workspaceId", "cellId", "content")},
    "required": ["notebookId", "workspaceId", "cellId", "content"],
}


def make_mcp(result=("ok", False), tools=None):
    mcp = MagicMock()
    mcp.list_tools = AsyncMock(
        return_value=tools if tools is not None else [{"name": "update_cell", "description": "d", "inputSchema": SCHEMA}]
    )
    mcp.call_tool = AsyncMock(return_value=result)
    return mcp


def tool_call(arguments: str, call_id="t1"):
    return {"content": None, "tool_calls": [{"id": call_id, "function": {"name": "update_cell", "arguments": arguments}}]}


def model_replies(mocker, *replies):
    """Patch the model call; `.sent` holds the messages as each call saw them (the list grows afterwards)."""
    queue = list(replies)

    class Spy(AsyncMock):
        sent: list = []

    async def fake(http, key, model, messages, tools=None, **kwargs):
        spy.sent.append(list(messages))
        return queue.pop(0), 1

    spy = Spy(side_effect=fake)
    spy.sent = []
    return mocker.patch.object(cell_edit, "call_model", new=spy)


@pytest.mark.asyncio
async def test_the_edit_is_applied_through_update_cell_with_the_ids_from_the_request(mocker):
    mcp = make_mcp()
    call = model_replies(mocker, tool_call('{"content": "select 2", "cellId": "WRONG", "notebookId": "WRONG"}'))

    written = await cell_edit.update_cell_with_mcp(mcp, "k", "m", "sys", "make it 2", CELL)

    assert written == "select 2"
    mcp.call_tool.assert_awaited_once_with(
        "update_cell", {"notebookId": "d1", "workspaceId": "w1", "cellId": "c1", "content": "select 2"}
    )
    # the model is made to call update_cell, and is offered nothing else
    assert call.await_args.kwargs["tool_choice"] == {"type": "function", "function": {"name": "update_cell"}}
    assert [t["function"]["name"] for t in call.await_args.args[4]] == ["update_cell"]


@pytest.mark.asyncio
async def test_a_call_with_no_content_is_sent_back_and_corrected(mocker):
    mcp = make_mcp()
    call = model_replies(mocker, tool_call("{}"), tool_call('{"content": "select 2"}', "t2"))

    written = await cell_edit.update_cell_with_mcp(mcp, "k", "m", "sys", "u", CELL)

    assert written == "select 2"
    assert call.await_count == 2
    fed_back = call.sent[1][-1]
    assert fed_back["role"] == "tool" and "content" in fed_back["content"]
    mcp.call_tool.assert_awaited_once()  # the bad call never reached the server


@pytest.mark.asyncio
async def test_a_server_error_goes_back_to_the_model_once(mocker):
    mcp = make_mcp()
    mcp.call_tool.side_effect = [("print() of a table is not allowed", True), ("ok", False)]
    call = model_replies(mocker, tool_call('{"content": "print(df)"}'), tool_call('{"content": "df"}', "t2"))

    written = await cell_edit.update_cell_with_mcp(mcp, "k", "m", "sys", "u", CELL)

    assert written == "df"
    assert "not allowed" in call.sent[1][-1]["content"]


@pytest.mark.asyncio
async def test_it_gives_up_after_two_tries_with_the_servers_reason(mocker):
    mcp = make_mcp(("cell is running", True))
    model_replies(mocker, tool_call('{"content": "a"}'), tool_call('{"content": "b"}', "t2"))

    with pytest.raises(CellEditFailed, match="cell is running"):
        await cell_edit.update_cell_with_mcp(mcp, "k", "m", "sys", "u", CELL)


@pytest.mark.asyncio
async def test_a_reply_without_a_tool_call_is_told_to_use_the_tool(mocker):
    mcp = make_mcp()
    call = model_replies(mocker, {"content": "select 2"}, tool_call('{"content": "select 2"}'))

    assert await cell_edit.update_cell_with_mcp(mcp, "k", "m", "sys", "u", CELL) == "select 2"
    assert "calling update_cell" in call.sent[1][-1]["content"]


@pytest.mark.asyncio
async def test_no_update_cell_tool_means_a_clear_failure(mocker):
    model = mocker.patch.object(cell_edit, "call_model", new=AsyncMock())

    with pytest.raises(CellEditFailed, match="no update_cell"):
        await cell_edit.update_cell_with_mcp(make_mcp(tools=[]), "k", "m", "sys", "u", CELL)
    model.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_transform_runs_on_the_content_before_it_is_written(mocker):
    mcp = make_mcp()
    model_replies(mocker, tool_call('{"content": "print(df)"}'))

    written = await cell_edit.update_cell_with_mcp(mcp, "k", "m", "sys", "u", CELL, transform=lambda text: text.upper())

    assert written == "PRINT(DF)"
    assert mcp.call_tool.await_args.args[1]["content"] == "PRINT(DF)"
