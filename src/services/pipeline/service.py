from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from src.config.settings import settings
from src.models.base import ChatContext
from src.services.completions.service import CompletionService
from src.services.completions.models import CompletionRequest, Message
from src.services.notebook_context.service import NotebookContextService
from src.services.notebook_context.models import AiContextRequest
from src.services.intent.service import ParseIntentService
from src.services.intent.models import ParseIntentRequest, Intent, IntentClass, ParsedIntent
from src.services.block_planner.service import PlanBlocksService
from src.services.block_planner.models import PlanBlocksRequest, BlockPlan


@dataclass
class PipelineState:
    messages: list[Message]
    model: str
    api_key: str
    context: ChatContext
    parsed_intent: ParsedIntent | None = None
    block_plan: BlockPlan | None = None
    notebook_markdown: str | None = None
    output: str | None = None

    @property
    def has_focused_blocks(self) -> bool:
        return bool(self.context.focused_block_ids)

    @property
    def has_non_explicit_focus(self) -> bool:
        return self.parsed_intent is not None and self.parsed_intent.references_block


async def node_parse_intent(state: PipelineState) -> PipelineState:
    user_message = next(m for m in reversed(state.messages) if m.role == "user")

    req = ParseIntentRequest(
        message=user_message.content,
        model=state.model,
        openrouter_api_key=state.api_key,
        history=state.messages[:-1],
        context=state.context,
    )

    raw = ""
    async for chunk in ParseIntentService(req).stream():
        raw += chunk

    try:
        data = json.loads(raw)
        state.parsed_intent = ParsedIntent(
            intent_class=IntentClass(data.get("intent_class", "analytical")),
            intent_status=data.get("intent_status", "error"),
            intent=data.get("intent"),
            references_block=data.get("references_block", False),
        )
    except (json.JSONDecodeError, ValueError):
        state.parsed_intent = ParsedIntent(
            intent_class=IntentClass.ANALYTICAL,
            intent_status="error",
            intent=None,
            references_block=False,
        )

    return state


async def node_fetch_context(state: PipelineState) -> PipelineState:
    ctx = AiContextRequest(
        document_id=state.context.document_id,
        workspace_id=state.context.workspace_id,
        focused_block_ids=list(state.context.focused_block_ids) if state.context.focused_block_ids else None,
    )

    state.notebook_markdown = await NotebookContextService(
        nest_base_url=settings.nest_base_url,
        intent=state.parsed_intent.intent_class,
        has_focused_blocks=state.has_focused_blocks,
        has_non_explicit_focus=state.has_non_explicit_focus,
    ).fetch(ctx)

    return state


async def node_plan_blocks(state: PipelineState) -> PipelineState:
    intent = Intent.model_validate(state.parsed_intent.intent)

    req = PlanBlocksRequest(
        message="",
        model=state.model,
        openrouter_api_key=state.api_key,
        intent=intent,
        context=state.context,
    )

    state.block_plan = await PlanBlocksService(req).plan()
    return state


async def node_complete(state: PipelineState) -> PipelineState:
    messages = state.messages

    if state.notebook_markdown:
        messages = [
            Message(role="system", content=f"<notebook>\n{state.notebook_markdown}\n</notebook>"),
            *messages,
        ]

    req = CompletionRequest(
        messages=messages,
        model=state.model,
        openrouter_api_key=state.api_key,
        context=state.context,
    )

    result = ""
    async for chunk in CompletionService().stream(req):
        result += chunk

    state.output = result
    return state


async def run_pipeline(state: PipelineState) -> PipelineState:
    state = await node_parse_intent(state)
    if not state.parsed_intent.is_complete:
        return state
    state = await node_fetch_context(state)
    if state.parsed_intent.intent_class == IntentClass.ANALYTICAL:
        state = await node_plan_blocks(state)
    state = await node_complete(state)
    return state