from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic_ai import Agent, Tool
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from src.config.settings import settings
from src.models.base import ChatContext, Message
from src.services.agent.intent import classify, narrow
from src.services.agent.mcp_client import McpClient, McpError
from src.services.agent.memory import attempt_for, prior_research, remember
from src.services.agent.tools import invalid_arguments, select_tools, to_openrouter_tool
from src.services.research_memory.models import Attempt
from src.util.cache import clear_active_job, clear_job_cancel, is_job_cancelled
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
            f"You are in notebook {context.document_id} of workspace {context.workspace_id}. Every tool already works on this notebook, so you never pass a notebook or workspace id. This notebook already exists and is open in front of the user: do all the work in it, by adding and editing its cells. Never create, copy or switch to another notebook or workspace, even for a new analysis.{focused}",
            "Use the tools to do the work instead of describing it, and never invent figures: report what the tools returned.",
            mcp_instructions,
            prior_research,
        ]
        if part
    )


def notebook_ids(context: ChatContext) -> dict[str, str]:
    """The notebook the chat is open in, under the names the MCP tools use for it."""
    return {"notebookId": context.document_id, "workspaceId": context.workspace_id}


def pinned_arguments(schema: dict[str, Any], context: ChatContext) -> dict[str, str]:
    """The ids this tool takes, set to the current notebook. They always come from the request, never from the model."""
    return {k: v for k, v in notebook_ids(context).items() if k in schema.get("properties", {})}


def without_pinned(spec: dict[str, Any], context: ChatContext) -> dict[str, Any]:
    """The tool as the model sees it: the notebook ids are gone, since it cannot choose them."""
    schema = spec["function"]["parameters"]
    pinned = pinned_arguments(schema, context)
    if not pinned:
        return spec
    shown = {
        **schema,
        "properties": {k: v for k, v in schema["properties"].items() if k not in pinned},
        "required": [k for k in schema.get("required", []) if k not in pinned],
    }
    return {**spec, "function": {**spec["function"], "parameters": shown}}


class JobCancelled(Exception):
    """The user stopped the chat while the agent was working."""


def _make_model(state: AgentState, http: httpx.AsyncClient) -> OpenRouterModel:
    """The user's own OpenRouter key and chosen model."""
    client = AsyncOpenAI(base_url=settings.openrouter_base_url, api_key=state.api_key, http_client=http)
    return OpenRouterModel(state.model, provider=OpenRouterProvider(openai_client=client))


def _history(messages: list[Message]) -> list[ModelMessage]:
    """The chat so far, as the harness's message types. The last user message is sent separately."""
    history: list[ModelMessage] = []
    for m in messages:
        if m.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=m.content)]))
        elif m.role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=m.content)]))
        elif m.role == "system":
            history.append(ModelRequest(parts=[SystemPromptPart(content=m.content)]))
    return history


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
        if name in schemas:
            # Every tool works on the notebook the chat is open in, whatever the model sent.
            arguments = {**arguments, **pinned_arguments(schemas[name], state.context)}
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


def _build_tools(
    specs: list[dict[str, Any]], schemas: dict[str, dict], mcp: McpClient, state: AgentState, envelope: StreamEnvelope
) -> list[Tool]:
    """One harness tool per MCP tool. Each call still goes through _run_tool_call, so argument
    checks, cards in the chat, progress lines and notebook memory work as before."""

    def make(name: str) -> Callable[..., Awaitable[str]]:
        async def call(**arguments: Any) -> str:
            if await is_job_cancelled(state.job_id):
                raise JobCancelled
            message = await _run_tool_call(
                {"id": str(uuid.uuid4()), "function": {"name": name, "arguments": json.dumps(arguments)}},
                schemas,
                mcp,
                state,
                envelope,
            )
            return message["content"]

        return call

    return [
        Tool.from_schema(
            make(spec["function"]["name"]),
            name=spec["function"]["name"],
            description=spec["function"].get("description", ""),
            json_schema=without_pinned(spec, state.context)["function"]["parameters"],
            sequential=True,  # notebook edits (add/update/delete cell) are order-dependent
        )
        for spec in specs
    ]


async def _loop(
    state: AgentState, envelope: StreamEnvelope, mcp: McpClient, http: httpx.AsyncClient, prior: str
) -> AgentState:
    if await is_job_cancelled(state.job_id):
        log.info("agent cancelled for chat_id=%s", state.context.chat_id)
        return state

    specs = [to_openrouter_tool(t) for t in select_tools(await mcp.list_tools())]
    intent = await classify(http, state.api_key, state.model, state.messages)
    log.info("agent intent=%s chat_id=%s", intent, state.context.chat_id)
    specs = narrow(specs, intent)
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in specs}

    last_user = next((i for i in range(len(state.messages) - 1, -1, -1) if state.messages[i].role == "user"), None)
    prompt = state.messages[last_user].content if last_user is not None else ""
    earlier = [m for i, m in enumerate(state.messages) if i != last_user]

    agent = Agent(
        _make_model(state, http),
        instructions=system_prompt(state.context, await mcp.instructions(), prior),
        tools=_build_tools(specs, schemas, mcp, state, envelope),
        model_settings=ModelSettings(
            temperature=settings.AGENT_TEMPERATURE, max_tokens=state.max_tokens, parallel_tool_calls=False
        ),
    )
    limits = UsageLimits(request_limit=settings.AGENT_MAX_STEPS, total_tokens_limit=settings.AGENT_MAX_TOKENS)

    try:
        result = await agent.run(prompt, message_history=_history(earlier), usage_limits=limits)
    except JobCancelled:
        log.info("agent cancelled for chat_id=%s", state.context.chat_id)
        return state
    except UsageLimitExceeded as exc:
        if "token" in str(exc).lower():
            return await _finish(
                state, envelope, "I hit the token limit for one request. Ask me to continue and I will pick up from here."
            )
        return await _finish(
            state, envelope, "I ran out of steps before finishing. Ask me to continue and I will pick up from here."
        )
    except ModelHTTPError as exc:
        if exc.status_code == 404 and "tool" in str(exc.body).lower():
            raise RuntimeError(
                f"The model {state.model} cannot call tools, which chat needs. Pick another model in the model menu."
            ) from exc
        raise
    return await _finish(state, envelope, result.output or "")


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
            if await is_job_cancelled(state.job_id):
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
        await clear_active_job(chat_id, state.job_id)
        await clear_job_cancel(state.job_id)
