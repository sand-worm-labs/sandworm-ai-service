from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.models.base import Message
from src.services.agent import intent


def tool(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


ALL = [tool(n) for n in ["get_notebook", "add_cell", "update_cell", "run_notebook", "get_dashboard", "search_tools"]]
MESSAGES = [Message(role="user", content="thanks!")]


@pytest.mark.parametrize(
    "text,expected",
    [("edit", "edit"), (" Chat.", "chat"), ("run: the notebook", "run"), ("banana", "general"), ("", "general")],
)
def test_parse_intent_takes_the_first_known_word_and_otherwise_is_unsure(text, expected):
    assert intent.parse_intent(text) == expected


def names(tools):
    return [t["function"]["name"] for t in tools]


def test_chat_gets_no_tools_and_edit_gets_only_editing_tools():
    assert intent.narrow(ALL, "chat") == []
    assert names(intent.narrow(ALL, "edit")) == ["get_notebook", "add_cell", "update_cell", "search_tools"]
    assert "run_notebook" not in names(intent.narrow(ALL, "edit"))


def test_an_unsure_label_keeps_every_allowed_tool():
    assert intent.narrow(ALL, "general") == ALL


@pytest.mark.asyncio
async def test_classify_reads_the_label_the_model_gives(mocker):
    mocker.patch.object(intent.settings, "AGENT_INTENT_ENABLED", True)
    call = mocker.patch.object(intent, "call_model", new=AsyncMock(return_value=({"content": "chat"}, 3)))

    assert await intent.classify(None, "k", "m", MESSAGES) == "chat"
    assert call.await_args.kwargs["max_tokens"] == 8


@pytest.mark.asyncio
async def test_classify_falls_back_to_general_when_the_call_fails_or_is_off(mocker):
    mocker.patch.object(intent.settings, "AGENT_INTENT_ENABLED", True)
    mocker.patch.object(intent, "call_model", new=AsyncMock(side_effect=RuntimeError("no")))
    assert await intent.classify(None, "k", "m", MESSAGES) == "general"

    mocker.patch.object(intent.settings, "AGENT_INTENT_ENABLED", False)
    assert await intent.classify(None, "k", "m", MESSAGES) == "general"
