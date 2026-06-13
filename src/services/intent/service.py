from __future__ import annotations

import json
import re
from typing import AsyncIterator

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.providers.openrouter import make_streaming_llm
from src.util.cache import publish_job_event

from src.models.base import ChatContext
from .models import IntentClass, ParseIntentRequest
from .prompts import CLASSIFIER_PROMPT, SYSTEM_PROMPTS, _ANALYTICAL_PROMPT

class ParseIntentService:
    def __init__(self, req: ParseIntentRequest):
        self.llm = make_streaming_llm(api_key=req.openrouter_api_key, model=req.model)
        self.req = req

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

    async def _parse(self, intent_class: IntentClass) -> str:
        prompt = SYSTEM_PROMPTS.get(intent_class, _ANALYTICAL_PROMPT)
        messages = [SystemMessage(content=prompt)]

        for turn in self.req.history:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            elif turn.role == "assistant":
                messages.append(AIMessage(content=turn.content))

        messages.append(HumanMessage(content=self.req.message))

        full = ""
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                full += chunk.content

        return re.sub(r"^```(?:json)?\s*|\s*```$", "", full.strip())

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
                parsed = await self._parse(intent_class)
                data   = json.loads(parsed)

                payload = {
                    "intent_class":     intent_class.value,
                    "intent_status":    "complete",
                    "references_block": references_block,
                    "intent":           data.get("intent"),
                }

                await self._publish("intent_parsed", payload)
                yield json.dumps(payload)
                return

            data   = json.loads(await self._parse(intent_class))
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
                payload["message"]   = data.get("message")
                payload["questions"] = data.get("questions", [])
                await self._publish("follow_up", payload)

            else:
                await self._publish("intent_error", payload)

            yield json.dumps(payload)

        except Exception as exc:
            error = {"status": "error", "detail": str(exc)}
            await self._publish("intent_error", error)
            yield json.dumps(error)