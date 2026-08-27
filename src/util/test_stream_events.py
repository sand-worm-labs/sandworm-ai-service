from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.util.stream_events import StreamEnvelope


@pytest.fixture
def publish(mocker):
    return mocker.patch("src.util.stream_events.publish_job_event", new_callable=AsyncMock)


@pytest.fixture
def envelope():
    return StreamEnvelope(job_id="job-1", chat_id="chat-1")


@pytest.mark.asyncio
async def test_message_start(envelope, publish):
    await envelope.message_start()

    publish.assert_awaited_once_with(
        "job-1",
        {"type": "message_start", "message": {"id": "job-1", "chat_id": "chat-1"}},
        "chat-1",
    )


@pytest.mark.asyncio
async def test_thinking_emits_start_delta_stop_at_index_zero(envelope, publish):
    await envelope.thinking("planning stuff", 42)

    assert publish.await_count == 3
    calls = [c.args[1] for c in publish.await_args_list]

    assert calls[0] == {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}
    assert calls[1] == {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "planning stuff", "duration_ms": 42},
    }
    assert calls[2] == {"type": "content_block_stop", "index": 0}


@pytest.mark.asyncio
async def test_block_generating_then_ready_reuse_same_index(envelope, publish):
    await envelope.block_generating("block-1", "sql", "Top holders")
    await envelope.block_ready("block-1", "sql", "Top holders", "SELECT 1")

    start_call = publish.await_args_list[0].args[1]
    delta_call = publish.await_args_list[1].args[1]
    stop_call = publish.await_args_list[2].args[1]

    assert start_call["index"] == 0
    assert start_call["content_block"]["action"] == "generating"

    assert delta_call["index"] == 0
    assert delta_call["delta"] == {
        "type": "block_action_delta",
        "action": "ran",
        "block_id": "block-1",
        "block_type": "sql",
        "block_title": "Top holders",
        "content": "SELECT 1",
        "data_source": None,
        "dataframe_name": None,
    }
    assert stop_call == {"type": "content_block_stop", "index": 0}


@pytest.mark.asyncio
async def test_block_ready_for_unknown_block_id_is_noop(envelope, publish):
    await envelope.block_ready("never-started", "sql", "x", "SELECT 1")

    publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_blocks_get_distinct_indices(envelope, publish):
    await envelope.block_generating("block-1", "sql", "A")
    await envelope.block_generating("block-2", "python", "B")

    first_start = publish.await_args_list[0].args[1]
    second_start = publish.await_args_list[1].args[1]

    assert first_start["index"] == 0
    assert second_start["index"] == 1


@pytest.mark.asyncio
async def test_text_delta_opens_block_once_and_reuses_index(envelope, publish):
    await envelope.text_delta("Hel")
    await envelope.text_delta("lo")

    calls = [c.args[1] for c in publish.await_args_list]

    assert calls[0] == {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    assert calls[1] == {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}
    # second token: no new content_block_start, straight to another delta at the same index
    assert calls[2] == {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_follow_up_rides_on_message_delta(envelope, publish):
    questions = [{"id": "q1", "text": "Which chain?"}]
    await envelope.follow_up("Need more info", questions)

    publish.assert_awaited_once_with(
        "job-1",
        {"type": "message_delta", "delta": {"follow_up": {"message": "Need more info", "questions": questions}}},
        "chat-1",
    )


@pytest.mark.asyncio
async def test_message_stop_closes_open_text_block(envelope, publish):
    await envelope.text_delta("hi")
    publish.reset_mock()

    await envelope.message_stop()

    calls = [c.args[1] for c in publish.await_args_list]
    assert calls == [
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]


@pytest.mark.asyncio
async def test_message_stop_without_open_text_block_skips_stop_event(envelope, publish):
    await envelope.message_stop()

    calls = [c.args[1] for c in publish.await_args_list]
    assert calls == [
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]


@pytest.mark.asyncio
async def test_error_emits_typed_error_event(envelope, publish):
    await envelope.error("intent_error", "Intent parsing failed")

    publish.assert_awaited_once_with(
        "job-1",
        {"type": "error", "error": {"type": "intent_error", "message": "Intent parsing failed"}},
        "chat-1",
    )


@pytest.mark.asyncio
async def test_chat_id_none_is_forwarded_untouched(publish):
    envelope = StreamEnvelope(job_id="job-2", chat_id=None)
    await envelope.message_start()

    publish.assert_awaited_once_with(
        "job-2",
        {"type": "message_start", "message": {"id": "job-2", "chat_id": None}},
        None,
    )
