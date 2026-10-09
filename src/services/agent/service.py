from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.config.settings import settings
from src.models.base import ChatContext, Message
from src.services.agent.mcp_client import McpClient, McpError
from src.services.agent.memory import attempt_for, prior_research, remember
from src.services.agent.model import call_model
from src.services.agent.tools import invalid_arguments, select_tools, to_openrouter_tool
from src.services.research_memory.models import Attempt
from src.util.cache import clear_active_job, is_job_cancelled
from src.util.stream_events import StreamEnvelope

log = logging.getLogger("sandworm.agent")

STREAM_CHUNK_CHARS = 40
CELL_ACTIONS = {"add_cell": "created", "update_cell": "edited"}


@dataclass
class AgentState:
    messages: list[Message]
    model: str
    api_key: str
    context: ChatContext
    # Configured on the Node side via AI_CHAT_TEMPERATURE/AI_CHAT_MAX_TOKENS.
    temperature: float = 0.7
    max_tokens: int | None = None
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    output: str | None = None
    # Every tool the AI tried this run, for notebook memory.
    attempts: list[Attempt] = field(default_factory=list)


def system_prompt(context: ChatContext, mcp_instructions: str, prior_research: str = "") -> str:
    # Positioning and the "suggest next steps" rule arrive in mcp_instructions:
    # the MCP server is the one place they are written.
    focused = f" The user has these blocks focused: {', '.join(context.focused_block_ids)}." if context.focused_block_ids else ""
    return " ".join(
        part
        for part in [
            "You are Sandworm's onchain analytics assistant, working for the user through Sandworm's tools.",
            f"You are in notebook {context.document_id} of workspace {context.workspace_id}; pass these ids to tools that take them.{focused}",
            "Use the tools to do the work instead of describing it, and never invent figures: report what the tools returned.",
            mcp_instructions,
            prior_research,
        ]
        if part
    )


async def _chat(
    http: httpx.AsyncClient, state: AgentState, messages: list[dict], tools: list[dict]
) -> tuple[dict[str, Any], int]:
    return await call_model(http, state.api_key, state.model, messages, tools, state.temperature, state.max_tokens)


async def _finish(state: AgentState, envelope: StreamEnvelope, text: str) -> AgentState:
    state.output = text
    for i in range(0, len(text), STREAM_CHUNK_CHARS):
        await envelope.text_delta(text[i : i + STREAM_CHUNK_CHARS])
    return state


async def emit_cell_card(envelope: StreamEnvelope, name: str, arguments: dict[str, Any], result: str) -> None:
    """Show an added or edited cell in the chat as a card."""
    try:
        cell = json.loads(result)["cell"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return
    block_id, kind = cell.get("id"), cell.get("kind", "")
    if not block_id:
        return
    title = cell.get("title") or arguments.get("title", "")
    await envelope.block_generating(block_id, kind, title)
    await envelope.block_ready(
        block_id,
        kind,
        title,
        arguments.get("content", ""),
        data_source=arguments.get("dataSource"),
        dataframe_name=arguments.get("dataframeName"),
        action=CELL_ACTIONS[name],
    )


async def _run_tool_call(
    call: dict[str, Any], schemas: dict[str, dict], mcp: McpClient, state: AgentState, envelope: StreamEnvelope
) -> dict[str, str]:
    """Run one tool call the model asked for and return the tool message to send back."""
    name = call["function"]["name"]
    started = time.monotonic()
    arguments: dict[str, Any] = {}
    try:
        arguments = json.loads(call["function"].get("arguments") or "{}")
        # Checked here so a malformed call never reaches the server; the
        # model gets the problem back and can retry.
        problem = invalid_arguments(schemas[name], arguments) if name in schemas else f"Unknown tool {name}."
        text, is_error = (problem, True) if problem else await mcp.call_tool(name, arguments)
    except (McpError, json.JSONDecodeError, httpx.HTTPError) as exc:
        text, is_error = f"Tool call failed: {exc}", True

    if attempt := attempt_for(name, arguments, text, is_error):
        state.attempts.append(attempt)
    if name in CELL_ACTIONS and not is_error:
        await emit_cell_card(envelope, name, arguments, text)
    await envelope.thinking(f"{'Failed' if is_error else 'Ran'} {name}", int((time.monotonic() - started) * 1000))
    return {"role": "tool", "tool_call_id": call["id"], "content": text}


async def _loop(
    state: AgentState, envelope: StreamEnvelope, mcp: McpClient, http: httpx.AsyncClient, prior: str
) -> AgentState:
    tools = [to_openrouter_tool(t) for t in select_tools(await mcp.list_tools())]
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(state.context, await mcp.instructions(), prior)},
        *({"role": m.role, "content": m.content} for m in state.messages),
    ]
    tokens_used = 0

    for _ in range(settings.AGENT_MAX_STEPS):
        if await is_job_cancelled(state.context.chat_id):
            log.info("agent cancelled for chat_id=%s", state.context.chat_id)
            return state
        if tokens_used >= settings.AGENT_MAX_TOKENS:
            return await _finish(
                state, envelope, "I hit the token limit for one request. Ask me to continue and I will pick up from here."
            )

        message, used = await _chat(http, state, messages, tools)
        tokens_used += used
        calls = message.get("tool_calls") or []
        if not calls:
            return await _finish(state, envelope, message.get("content") or "")

        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
        for call in calls:
            messages.append(await _run_tool_call(call, schemas, mcp, state, envelope))

    return await _finish(
        state, envelope, "I ran out of steps before finishing. Ask me to continue and I will pick up from here."
    )


async def run_agent(
    state: AgentState, envelope: StreamEnvelope, mcp: McpClient, http: httpx.AsyncClient | None = None
) -> AgentState:
    """Tool-calling loop over the MCP server, as the user whose token arrived with the request.

    Bounded three ways per run: steps, total model tokens, and wall-clock time.
    """
    http = http or httpx.AsyncClient(timeout=180)
    try:
        question = next((m.content for m in reversed(state.messages) if m.role == "user"), "")
        prior = await prior_research(state.context.workspace_id, state.context.document_id, question)
        async with asyncio.timeout(settings.AGENT_MAX_SECONDS):
            return await _loop(state, envelope, mcp, http, prior)
    except TimeoutError:
        log.warning("agent hit the %ss limit for chat_id=%s", settings.AGENT_MAX_SECONDS, state.context.chat_id)
        return await _finish(
            state, envelope, "I hit the time limit for one request. Ask me to continue and I will pick up from here."
        )
    finally:
        await http.aclose()


async def run_chat(state: AgentState) -> AgentState:
    """One chat turn. Every notebook read and write goes through the MCP server."""
    chat_id = state.context.chat_id
    envelope = StreamEnvelope(job_id=state.job_id, chat_id=chat_id)
    try:
        await envelope.message_start()

        if not state.context.user_token:
            await envelope.error("error", "Chat needs the signed-in user's token to reach the notebook tools.")
            return state

        mcp = McpClient(settings.MCP_URL, state.context.user_token)
        try:
            if await is_job_cancelled(chat_id):
                log.info("chat cancelled before start for chat_id=%s", chat_id)
            else:
                state = await run_agent(state, envelope, mcp)

            # Without message_stop the SSE stream never terminates and the
            # frontend's isLoading flag gets stuck.
            await envelope.message_stop()
            # After the answer is out, so storing the notebook never slows the reply.
            await remember(state.context.workspace_id, state.context.document_id, state.job_id, state.attempts, mcp)
        finally:
            await mcp.aclose()
        return state
    except Exception as exc:
        await envelope.error("error", str(exc))
        raise
    finally:
        await clear_active_job(chat_id)
