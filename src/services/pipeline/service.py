from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from src.config.settings import settings
from src.models.base import ChatContext
from src.util.cache import clear_active_job, clear_job_cancel, is_job_cancelled
from src.util.llm_json import LLMJSONError, parse_json_object
from src.util.stream_events import StreamEnvelope
from src.services.completions.service import CompletionService
from src.services.completions.models import CompletionRequest, Message
from src.services.notebook_context.service import NotebookContextService
from src.services.notebook_context.models import AiContextRequest
from src.services.intent.service import ParseIntentService
from src.services.intent.models import ParseIntentRequest, Intent, IntentClass, ParsedIntent
from src.services.block_planner.service import PlanBlocksService
from src.services.block_planner.models import PlanBlocksRequest, BlockPlan
from src.services.block_action.service import BlockActionService
from src.services.block_action.model import GeneratedBlock

log = logging.getLogger("sandworm.pipeline")


@dataclass
class PipelineState:
    messages: list[Message]
    model: str
    api_key: str
    context: ChatContext
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parsed_intent: ParsedIntent | None = None
    block_plan: BlockPlan | None = None
    generated_blocks: list[GeneratedBlock] | None = None
    notebook_markdown: str | None = None
    output: str | None = None

    @property
    def has_focused_blocks(self) -> bool:
        return bool(self.context.focused_block_ids)

    @property
    def has_non_explicit_focus(self) -> bool:
        return self.parsed_intent is not None and self.parsed_intent.references_block


async def node_parse_intent(state: PipelineState, envelope: StreamEnvelope) -> PipelineState:
    user_message = next(m for m in reversed(state.messages) if m.role == "user")

    req = ParseIntentRequest(
        message=user_message.content,
        model=state.model,
        openrouter_api_key=state.api_key,
        history=state.messages[:-1],
        context=state.context,
        job_id=state.job_id,
    )

    raw = ""
    async for chunk in ParseIntentService(req, envelope).stream():
        raw += chunk

    try:
        # Pre-parse (strip fences/prose, confirm it's a JSON object) before
        # trusting its shape.
        data = parse_json_object(raw)
        state.parsed_intent = ParsedIntent(
            intent_class=IntentClass(data.get("intent_class", "analytical")),
            intent_status=data.get("intent_status", "error"),
            intent=data.get("intent"),
            references_block=data.get("references_block", False),
        )
    except (LLMJSONError, ValueError) as exc:
        log.warning("intent parse not valid JSON (%s); raw=%r", exc, raw[:2000])
        state.parsed_intent = ParsedIntent(
            intent_class=IntentClass.ANALYTICAL,
            intent_status="error",
            intent=None,
            references_block=False,
        )

    return state


async def node_fetch_notebook_context(state: PipelineState) -> PipelineState:
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


async def node_generate_blocks(state: PipelineState, envelope: StreamEnvelope) -> PipelineState:
    intent = Intent.model_validate(state.parsed_intent.intent)
    chat_id = state.context.chat_id if isinstance(state.context, ChatContext) else None
    service = BlockActionService(api_key=state.api_key, model=state.model, envelope=envelope)
    state.generated_blocks = await service.generate_blocks(
        state.block_plan, intent, state.notebook_markdown, chat_id=chat_id,
    )
    return state


async def node_complete(state: PipelineState, envelope: StreamEnvelope, chat_id: str | None) -> PipelineState:
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
    chunk_count = 0
    async for chunk in CompletionService().stream(req):
        # Checking Redis on every token would add real latency to the
        # stream — every 20 chunks is frequent enough to feel responsive to
        # an abort without hammering it.
        chunk_count += 1
        if chat_id is not None and chunk_count % 20 == 0 and await is_job_cancelled(chat_id):
            log.info("pipeline cancelled mid-completion for chat_id=%s", chat_id)
            state.output = result
            return state

        result += chunk
        await envelope.text_delta(chunk)

    state.output = result
    return state


async def _stop_if_cancelled(chat_id: str | None, envelope: StreamEnvelope, where: str) -> bool:
    if chat_id is None or not await is_job_cancelled(chat_id):
        return False

    log.info("pipeline cancelled at %s for chat_id=%s", where, chat_id)
    await envelope.message_stop()
    return True


async def run_pipeline(state: PipelineState) -> PipelineState:
    job_id = state.job_id
    chat_id = state.context.chat_id if isinstance(state.context, ChatContext) else None
    envelope = StreamEnvelope(job_id=job_id, chat_id=chat_id)
    try:
        await envelope.message_start()

        if await _stop_if_cancelled(chat_id, envelope, "start"):
            return state

        state = await node_parse_intent(state, envelope)
        if not state.parsed_intent.is_complete:
            # A follow-up question still ends this turn — without message_stop
            # the SSE stream never terminates, so the frontend's isLoading flag
            # gets stuck and blocks the user's follow-up answer from sending.
            # Safe to call even if an "error" event already ended the stream:
            # the Node side no-ops on message_stop once the subject is gone.
            await envelope.message_stop()
            return state

        if await _stop_if_cancelled(chat_id, envelope, "after intent parsing"):
            return state

        state = await node_fetch_notebook_context(state)

        if state.parsed_intent.intent_class in (IntentClass.ANALYTICAL, IntentClass.EDITORIAL):
            if await _stop_if_cancelled(chat_id, envelope, "after notebook context"):
                return state

            planning_start = time.monotonic()
            state = await node_plan_blocks(state)
            duration_ms = int((time.monotonic() - planning_start) * 1000)
            block_summaries = ", ".join(f"{b.type}: {b.title}" for b in state.block_plan.blocks)
            thinking = (
                f"Planning {len(state.block_plan.blocks)} block(s): {block_summaries}"
                if state.block_plan.blocks
                else "Couldn't plan any blocks for this — answering directly instead"
            )
            await envelope.thinking(thinking, duration_ms)

            if await _stop_if_cancelled(chat_id, envelope, "after planning"):
                return state

            state = await node_generate_blocks(state, envelope)

            if await _stop_if_cancelled(chat_id, envelope, "after block generation"):
                return state

        state = await node_complete(state, envelope, chat_id)
        await envelope.message_stop()
        return state

    except Exception as exc:
        await envelope.error("error", str(exc))
        raise
    finally:
        if chat_id is not None:
            await clear_active_job(chat_id)
            await clear_job_cancel(chat_id)