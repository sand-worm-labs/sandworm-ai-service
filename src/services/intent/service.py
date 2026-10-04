from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.providers.openrouter import make_streaming_llm
from src.util.cache import publish_job_event
from src.util.llm_json import LLMJSONError, parse_json_object
from src.util.stream_events import StreamEnvelope

from src.models.base import ChatContext
from src.services.open_data.service import FREE_PLAN_INTENT_NOTE
from .models import IntentClass, ParseIntentRequest
from .prompts import CLASSIFIER_PROMPT, SYSTEM_PROMPTS, _ANALYTICAL_PROMPT

log = logging.getLogger("sandworm.intent")

# Mirrors block_planner's retry pattern — an LLM empty/malformed response is
# usually a one-off streaming hiccup, not a genuine incapability, so it's
# worth one corrective retry before surfacing an error to the user.
MAX_PARSE_ATTEMPTS = 2

class ParseIntentService:
    def __init__(self, req: ParseIntentRequest, envelope: StreamEnvelope | None = None):
        self.llm = make_streaming_llm(api_key=req.openrouter_api_key, model=req.model)
        self.req = req
        self.envelope = envelope

    async def _classify(self) -> tuple[IntentClass, bool]:
        result = ""
        async for chunk in self.llm.astream([
            SystemMessage(content=CLASSIFIER_PROMPT),
            HumanMessage(content=self.req.message),
        ]):
            if chunk.content:
                result += chunk.content

        parts = result.strip().lower().split("|")
        try:
            intent_class = IntentClass(parts[0].strip())
        except ValueError:
            intent_class = IntentClass.ANALYTICAL

        return intent_class, len(parts) > 1 and parts[1].strip() == "yes"

    def _build_messages(self, intent_class: IntentClass) -> list:
        prompt = SYSTEM_PROMPTS.get(intent_class, _ANALYTICAL_PROMPT)
        messages: list = [SystemMessage(content=prompt)]
        if self.req.context.paid_plan_required:
            messages.append(SystemMessage(content=FREE_PLAN_INTENT_NOTE))

        for turn in self.req.history:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            elif turn.role == "assistant":
                messages.append(AIMessage(content=turn.content))

        messages.append(HumanMessage(content=self.req.message))
        return messages

    async def _stream_completion(self, messages: list) -> str:
        full = ""
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                full += chunk.content
        return full.strip()

    async def _parse_json(self, intent_class: IntentClass) -> dict:
        messages = self._build_messages(intent_class)
        last_exc: LLMJSONError | None = None
        raw = ""

        for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
            raw = await self._stream_completion(messages)
            try:
                return parse_json_object(raw)
            except LLMJSONError as exc:
                last_exc = exc
                log.warning(
                    "intent parse not valid JSON on attempt %d/%d (%s); raw=%r",
                    attempt, MAX_PARSE_ATTEMPTS, exc, raw[:2000],
                )
                if attempt == MAX_PARSE_ATTEMPTS:
                    break

                # Same self-correction trick as block_planner: hand the model
                # its own bad output back with a precise instruction rather
                # than starting over from scratch.
                messages = [
                    *messages,
                    AIMessage(content=raw),
                    HumanMessage(
                        content=(
                            "That response was not valid JSON matching the schema. "
                            "Return ONLY the JSON object — no markdown fences, no explanation."
                        )
                    ),
                ]

        assert last_exc is not None
        raise last_exc

    def _is_followup_clarification(self) -> bool:
        for turn in reversed(self.req.history):
            if turn.role == "assistant":
                try:
                    data = json.loads(turn.content)
                    if "intent_status" in data:
                        return data["intent_status"] == "clarify"
                except (json.JSONDecodeError, ValueError):
                    continue
        return False

    def _intent_class_from_history(self) -> IntentClass:
        for turn in reversed(self.req.history):
            if turn.role == "assistant":
                try:
                    data = json.loads(turn.content)
                    if "intent_class" in data:
                        return IntentClass(data["intent_class"])
                except (json.JSONDecodeError, ValueError):
                    continue
        return IntentClass.ANALYTICAL

    def _references_block_from_history(self) -> bool:
        for turn in reversed(self.req.history):
            if turn.role == "assistant":
                try:
                    data = json.loads(turn.content)
                    if "references_block" in data:
                        return bool(data["references_block"])
                except (json.JSONDecodeError, ValueError):
                    continue
        return False

    @property
    def _chat_id(self) -> str | None:
        return self.req.context.chat_id if isinstance(self.req.context, ChatContext) else None

    async def _publish(self, event_type: str, payload: dict) -> None:
        if self.req.job_id:
            await publish_job_event(self.req.job_id, {"type": event_type, **payload}, self._chat_id)

    async def stream(self) -> AsyncIterator[str]:
        try:
            is_first    = len(self.req.history) == 0
            is_followup = self._is_followup_clarification()

            if is_first or not is_followup:
                intent_class, references_block = await self._classify()
            else:
                intent_class     = self._intent_class_from_history()
                references_block = self._references_block_from_history()

            await self._publish("intent_classified", {
                "intent_class":     intent_class.value,
                "references_block": references_block,
            })

            if intent_class != IntentClass.ANALYTICAL:
                data = await self._parse_json(intent_class)

                payload = {
                    "intent_class":     intent_class.value,
                    "intent_status":    "complete",
                    "references_block": references_block,
                    "intent":           data.get("intent"),
                }

                await self._publish("intent_parsed", payload)
                yield json.dumps(payload)
                return

            data   = await self._parse_json(intent_class)
            status = data.get("status", "error")

            payload: dict = {
                "intent_class":     intent_class.value,
                "intent_status":    status,
                "references_block": references_block,
            }

            if status == "complete":
                payload["intent"] = data.get("intent")
                await self._publish("intent_parsed", payload)

            elif status == "clarify":
                message   = data.get("message")
                questions = data.get("questions", [])
                payload["message"]   = message
                payload["questions"] = questions
                if self.envelope:
                    await self.envelope.follow_up(message, questions)

            else:
                if self.envelope:
                    await self.envelope.error("intent_error", "Intent parsing failed")

            yield json.dumps(payload)

        except Exception as exc:
            error = {"status": "error", "message": str(exc)}
            if self.envelope:
                await self.envelope.error("intent_error", str(exc))
            yield json.dumps(error)