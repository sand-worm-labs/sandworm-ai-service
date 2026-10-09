from __future__ import annotations

import json

import httpx
import pytest

from src.services.agent import model


def mock_openrouter(mocker, reply: dict, status: int = 200):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(status, json=reply)

    real = httpx.AsyncClient
    mocker.patch.object(model.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return seen


@pytest.mark.asyncio
async def test_ask_sends_one_system_and_one_user_message_with_the_users_key(mocker):
    seen = mock_openrouter(mocker, {"choices": [{"message": {"content": "  hello \n"}}]})

    answer = await model.ask("sk-user", "some/model", "be brief", "hi", temperature=0.2, max_tokens=50)

    assert answer == "hello"
    assert seen["auth"] == "Bearer sk-user"
    assert seen["payload"] == {
        "model": "some/model",
        "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "max_tokens": 50,
    }


@pytest.mark.asyncio
async def test_ask_never_sends_tools(mocker):
    seen = mock_openrouter(mocker, {"choices": [{"message": {"content": "x"}}]})

    await model.ask("k", "m", "s", "u")

    assert "tools" not in seen["payload"] and "parallel_tool_calls" not in seen["payload"]


@pytest.mark.asyncio
async def test_a_failed_call_raises(mocker):
    mock_openrouter(mocker, {"error": {"message": "bad key"}}, status=401)

    with pytest.raises(httpx.HTTPStatusError):
        await model.ask("k", "m", "s", "u")
