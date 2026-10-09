from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.base import DocumentContext
from src.services.notebook_title import service as title

CONTEXT = DocumentContext(user_id="u", workspace_id="w1", document_id="d1", user_token="tok")


@pytest.fixture
def mcp(mocker):
    client = MagicMock()
    client.aclose = AsyncMock()
    client.call_tool = AsyncMock(return_value=('{"title": "Old", "content": "select tvl from uniswap"}', False))
    mocker.patch.object(title, "McpClient", return_value=client)
    return client


@pytest.mark.asyncio
async def test_rename_reads_the_notebook_and_has_the_mcp_apply_the_title(mcp, mocker):
    rename = mocker.patch.object(title, "rename_notebook_with_mcp", new=AsyncMock(return_value="Uniswap TVL"))

    assert await title.NotebookTitleService("k", "m", CONTEXT).rename() == "Uniswap TVL"

    mcp.call_tool.assert_awaited_once_with("get_notebook", {"notebookId": "d1", "workspaceId": "w1"})
    args = rename.await_args.args
    assert "Current title: Old" in args[4] and "select tvl from uniswap" in args[4]
    assert args[5:] == ("w1", "d1")
    mcp.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_notebook_that_cannot_be_read_still_gets_a_title(mcp, mocker):
    mcp.call_tool.side_effect = RuntimeError("down")
    rename = mocker.patch.object(title, "rename_notebook_with_mcp", new=AsyncMock(return_value="Fresh"))

    assert await title.NotebookTitleService("k", "m", CONTEXT).rename() == "Fresh"
    assert "(none)" in rename.await_args.args[4]


@pytest.mark.asyncio
async def test_rename_needs_the_users_token():
    context = CONTEXT.model_copy(update={"user_token": None})

    with pytest.raises(ValueError, match="token"):
        await title.NotebookTitleService("k", "m", context).rename()
