from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.base import ChatContext
from src.services.agent import service as agent
from src.services.completions.models import Message


def make_state(token="tok") -> agent.AgentState:
    return agent.AgentState(
        messages=[Message(role="user", content="hi")],
        model="m",
        api_key="k",
        context=ChatContext(user_id="u", workspace_id="w", document_id="d", chat_id="c", user_token=token),
        job_id="j",
    )


@pytest.fixture
def run(mocker):
    envelope = MagicMock()
    for name in ("message_start", "message_stop", "error"):
        setattr(envelope, name, AsyncMock())
    mocker.patch.object(agent, "StreamEnvelope", return_value=envelope)
    mcp = MagicMock(aclose=AsyncMock())
    mocker.patch.object(agent, "McpClient", return_value=mcp)
    return MagicMock(
        envelope=envelope,
        mcp=mcp,
        agent=mocker.patch.object(agent, "run_agent", new=AsyncMock(side_effect=lambda state, *_a, **_k: state)),
        remember=mocker.patch.object(agent, "remember", new=AsyncMock()),
        cancelled=mocker.patch.object(agent, "is_job_cancelled", new=AsyncMock(return_value=False)),
        clear_job=mocker.patch.object(agent, "clear_active_job", new=AsyncMock()),
        clear_cancel=mocker.patch.object(agent, "clear_job_cancel", new=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_a_chat_runs_stops_the_stream_remembers_and_clears_both_flags(run):
    await agent.run_chat(make_state())

    run.envelope.message_start.assert_awaited_once()
    run.agent.assert_awaited_once()
    run.envelope.message_stop.assert_awaited_once()
    run.remember.assert_awaited_once()
    run.mcp.aclose.assert_awaited_once()
    run.clear_job.assert_awaited_once_with("c", "j")
    # a Stop must not carry over to the next message in this chat
    run.clear_cancel.assert_awaited_once_with("j")


@pytest.mark.asyncio
async def test_a_stopped_chat_still_ends_the_stream_and_clears_the_stop(run):
    run.cancelled.return_value = True

    await agent.run_chat(make_state())

    run.agent.assert_not_awaited()
    run.envelope.message_stop.assert_awaited_once()
    run.clear_cancel.assert_awaited_once_with("j")


@pytest.mark.asyncio
async def test_without_a_token_the_user_gets_an_error_and_the_flags_are_still_cleared(run):
    await agent.run_chat(make_state(token=None))

    run.envelope.error.assert_awaited_once()
    run.agent.assert_not_awaited()
    run.clear_job.assert_awaited_once_with("c", "j")
    run.clear_cancel.assert_awaited_once_with("j")


@pytest.mark.asyncio
async def test_a_crash_is_reported_and_the_flags_are_still_cleared(run):
    run.agent.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await agent.run_chat(make_state())

    run.envelope.error.assert_awaited_once()
    run.mcp.aclose.assert_awaited_once()
    run.clear_job.assert_awaited_once_with("c", "j")
    run.clear_cancel.assert_awaited_once_with("j")
