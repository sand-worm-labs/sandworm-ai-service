from __future__ import annotations

import json
import re
from typing import AsyncIterator

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.providers.openrouter import make_streaming_llm
from src.util.cache import publish_job_event

from .models import IntentClass, ParseIntentRequest, ParsedIntent


CLASSIFIER_PROMPT = """Blockchain analytics notebook classifier.
Output: <class>|<references_block>
Classes: analytical|conversational|explanatory|editorial
- analytical: new query/analysis/viz
- editorial: modify/fix existing (default if unsure)
- explanatory: explain block or concept
- conversational: general question, no data op
references_block=yes → user references a notebook cell ("this block","that query","it")
references_block=no  → onchain ref ("block 18500000","0xabc...") or none
One line only."""

_ANALYTICAL_PROMPT = """Blockchain analytics intent parser. Scan history — never re-ask known params.

NON-NEGOTIABLES (in order):
1. No address + no protocol → ask for one
2. 0x... + no chain → ask chain
3. Multiple addresses, no chain → ask each

Address rules: ENS→ethereum no ask. Named entity→ask address. Chain in history→apply, never re-ask.
Deeper: wallet vs collection? comparison vs single? event anchor? multi-chain unified or separate (default unified)?
NEVER ask: timeframe precision, sub-entity selection, methodology, sensible defaults.
Vague protocol→ask. CEX name→flag+clarify. Token=protocol→ask which. Relative time→ask block/date.
"full report/deep dive"→output_scope=full. "quick/summary"→output_scope=summary. default=full.
sub_goals feasible=false only if provably requires off-chain data. Batch ALL questions into ONE follow_up.

CLARIFY: {"status":"clarify","type":"follow_up","message":"...","questions":[{"id":"...","text":"...","input_type":"radio|text|select","options":[{"label":"...","value":"..."}],"placeholder":"...","required":true}]}
COMPLETE: {"status":"complete","intent":{"goal":"...","entity":{"addresses":[{"address":"0x...","chain":"ethereum"}],"protocol_names":[]},"params":{},"sub_goals":[{"goal":"...","feasible":true,"reason":null}]}}
JSON only."""

_EDITORIAL_PROMPT = """Notebook edit parser.
Output ONLY: {"status":"complete","intent":{"goal":"<snake_case>","target":"<ref|null>","instruction":"<verbatim>"}}
JSON only."""

_EXPLANATORY_PROMPT = """Notebook explain parser.
Output ONLY: {"status":"complete","intent":{"goal":"explain","target":"<what|null>","question":"<verbatim>"}}
JSON only."""

_CONVERSATIONAL_PROMPT = """Notebook assistant.
Output ONLY: {"status":"complete","intent":{"goal":"answer","question":"<verbatim>"}}
JSON only."""

SYSTEM_PROMPTS: dict[IntentClass, str] = {
    IntentClass.ANALYTICAL:    _ANALYTICAL_PROMPT,
    IntentClass.EDITORIAL:     _EDITORIAL_PROMPT,
    IntentClass.EXPLANATORY:   _EXPLANATORY_PROMPT,
    IntentClass.CONVERSATIONAL:_CONVERSATIONAL_PROMPT,
}

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

    async def _publish(self, event_type: str, payload: dict) -> None:
        if self.req.job_id:
            await publish_job_event(self.req.job_id, {"type": event_type, **payload})

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