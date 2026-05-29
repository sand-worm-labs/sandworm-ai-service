from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config.settings import settings
from src.models.base import ChatContext, DocumentContext
from src.services.completions.service import CompletionService
from src.services.notebook_context.service import NotebookContextService
from src.services.intent.service import ParseIntentService
from src.services.intent.models import ParseIntentRequest
from src.services.completions.models import CompletionRequest, Message
from src.services.notebook_context.models import AiContextRequest


@dataclass
class PipelineState:
    messages: list[Message]
    model: str
    api_key: str
    context: ChatContext
    intent: dict[str, Any] | None = None
    notebook_markdown: str | None = None
    output: str | None = None


async def node_parse_intent(state: PipelineState) -> PipelineState:
    user_message = next(m for m in reversed(state.messages) if m.role == "user")
    req = ParseIntentRequest(
        message=user_message.content,
        model=state.model,
        openrouter_api_key=state.api_key,
        history=state.messages[:-1],
        context=state.context,
    )
    result = ""
    async for chunk in ParseIntentService(req).stream():
        result += chunk
    state.intent = result
    return state


async def node_fetch_context(state: PipelineState) -> PipelineState:
    ctx = AiContextRequest(
        document_id=state.context.document_id,
        workspace_id=state.context.workspace_id,
    )
    state.notebook_markdown = await NotebookContextService(settings.nest_base_url).fetch(ctx)
    return state


async def node_complete(state: PipelineState) -> PipelineState:
    messages = state.messages
    if state.notebook_markdown:
        messages = [Message(role="system", content=f"<notebook>\n{state.notebook_markdown}\n</notebook>"), *messages]

    input = CompletionRequest(messages=messages, model=state.model, openrouter_api_key=state.api_key)
    result = ""
    async for chunk in CompletionService().stream(input):
        result += chunk
    state.output = result
    return state


async def run_pipeline(state: PipelineState) -> PipelineState:
    state = await node_parse_intent(state)
    state = await node_fetch_context(state)
    state = await node_complete(state)
    return state