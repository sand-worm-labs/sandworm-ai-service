from __future__ import annotations

import json
import re
from typing import AsyncIterator

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .models import IntentClass, ParseIntentRequest, ParsedIntent


CLASSIFIER_PROMPT = """Blockchain analytics notebook intent classifier.

Output: <class>|<references_block>

Classes: analytical|conversational|explanatory|editorial
- analytical: new queries/analysis/viz
- editorial: modify/fix/optimize existing (default when unsure)
- explanatory: explain existing block or concept
- conversational: general question, no data op

"block" = notebook cell if: "this block","block 2","that query","it" → references_block=yes
"block" = onchain if: "block 18500000","0xabc..." → analytical, references_block=no

references_block=yes if references existing notebook cell.

analytical|no / editorial|yes / analytical|yes / conversational|no

One line only."""


SYSTEM_PROMPT = """Blockchain analytics intent parser for Sandworm.

Scan history first — never re-ask established params. Infer when possible.

NON-NEGOTIABLES (resolve in order):
1. No address + no protocol → ask for one
2. EVM address (0x...) + no chain → ask chain
3. Multiple addresses, no chain → ask each

Address rules:
- 0x... no chain → ask chain
- ENS (x.eth) → chain=ethereum, no ask
- Named entity → ask for explicit address
- Chain established in history → apply, never re-ask

Deeper (after non-negotiables):
- Wallet-level or collection-level?
- Comparison or single entity?
- Event/date anchor?
- Multi-chain: unified or separate? (default: unified)

NEVER ask: timeframe precision, sub-entity selection, methodology, sensible defaults.

Other rules:
- Vague protocol → ask which
- CEX name → flag + clarify
- Token=protocol name (e.g. USDC) → ask which
- Relative time anchor → ask date/block number
- "full report/deep dive" → output_scope=full; "quick/summary" → output_scope=summary; default=full

sub_goals: feasible=false only if provably requires off-chain data.
Batch ALL questions into ONE follow_up. Never clarify twice.

CLARIFY:
{"status":"clarify","type":"follow_up","message":"...","questions":[{"id":"...","text":"...","input_type":"radio|text|select","options":[{"label":"...","value":"..."}],"placeholder":"...","required":true}]}

radio/select = bounded. text = free input. Omit options for text.

COMPLETE:
{"status":"complete","intent":{"goal":"...","entity":{"addresses":[{"address":"0x...","chain":"ethereum"}],"protocol_names":[]},"params":{},"sub_goals":[{"goal":"...","feasible":true,"reason":null}]}}

JSON only. No markdown."""


SYSTEM_PROMPTS: dict[IntentClass, str] = {
    IntentClass.ANALYTICAL: SYSTEM_PROMPT,

    IntentClass.EDITORIAL: """Notebook intent parser. User wants to edit a block.
Output ONLY: {"status":"complete","intent":{"goal":"<snake_case>","target":"<ref or null>","instruction":"<verbatim>"}}
JSON only. No questions.""",

    IntentClass.EXPLANATORY: """Notebook intent parser. User wants an explanation.
Output ONLY: {"status":"complete","intent":{"goal":"explain","target":"<what or null>","question":"<verbatim>"}}
JSON only. No questions.""",

    IntentClass.CONVERSATIONAL: """Notebook assistant. Answer directly.
Output ONLY: {"status":"complete","intent":{"goal":"answer","question":"<verbatim>"}}
JSON only.""",
}


class ParseIntentService:
    def __init__(self, req: ParseIntentRequest):
        self.llm = ChatOpenRouter(
            api_key=req.openrouter_api_key,
            model=req.model,
            temperature=0,
            streaming=True,
        )
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
        prompt = SYSTEM_PROMPTS.get(intent_class, SYSTEM_PROMPT)
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

    async def stream(self) -> AsyncIterator[str]:
        try:
            is_first    = len(self.req.history) == 0
            is_followup = self._is_followup_clarification()

            if is_first or not is_followup:
                intent_class, references_block = await self._classify()
            else:
                intent_class     = self._intent_class_from_history()
                references_block = self._references_block_from_history()

            if intent_class != IntentClass.ANALYTICAL:
                parsed       = await self._parse(intent_class)
                data         = json.loads(parsed)
                yield json.dumps({
                    "intent_class":     intent_class.value,
                    "intent_status":    "complete",
                    "references_block": references_block,
                    "intent":           data.get("intent"),
                })
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
            elif status == "clarify":
                payload["message"]   = data.get("message")
                payload["questions"] = data.get("questions", [])

            yield json.dumps(payload)

        except Exception as exc:
            yield json.dumps({"status": "error", "detail": str(exc)})