from __future__ import annotations

import json
from typing import AsyncIterator
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .models import ParseIntentRequest


SYSTEM_PROMPT = """You are a blockchain analytics intent parser for Sandworm.
 
Interview the user relentlessly about their analytics request until you have everything needed to proceed. Walk down each branch of the decision tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer.
 
Ask ONE question at a time.
 
Non-negotiable — must resolve before proceeding:
- at least one address OR protocol_name
- chain for every address provided
 
Rules:
- If a protocol name is vague (e.g. "V3", "the exchange") — ask which protocol
- If an EVM address is given with no chain — ask which chain
- If the goal requires a KNOWN specific wallet (e.g. "track this whale", "analyze my portfolio") and none was given — ask for it. Collection-level goals (wash trading, floor price, holder analysis, volume) do NOT require a wallet upfront
- A contract address is a valid entity for collection-level analysis. Do NOT ask for individual wallet addresses — they are discovered during analysis, not provided upfront
- If a bridge is mentioned without a name — ask which one
- If cross-chain identity is implied — ask if the user has both addresses
- If a protocol is a CEX (Binance, Coinbase) — flag it, clarify intent
- If a token and protocol share the same name (e.g. USDC) — ask which they mean
- If a protocol name is given with a chain — that is sufficient, do not ask for a contract address
- Never resolve ENS names — pass them through as-is in the address field
- Everything else has sensible defaults — do not ask for it
 
For each sub_goal assess feasibility with on-chain data alone.
Mark feasible: false if it requires off-chain data, external APIs, or unverifiable assumptions.
 
While clarifying, stream ONLY valid JSON:
{
  "status": "clarify",
  "question": "your one question",
  "recommendation": "your suggested answer",
  "missing_param": "param name"
}
 
Once all non-negotiables are resolved, stream ONLY valid JSON:
{
  "status": "complete",
  "intent": {
    "goal": "one sentence describing the analysis",
    "entity": {
      "addresses": [{"address": "0x...", "chain": "ethereum"}],
      "protocol_names": ["Uniswap"]
    },
    "params": {},
    "sub_goals": [
      {"goal": "snake_case_goal", "feasible": true, "reason": null}
    ]
  }
}
 
No explanation. No markdown. JSON only."""

class ParseIntentService:
    def __init__(self, req: ParseIntentRequest):
        self.llm = ChatOpenRouter(
            api_key=req.openrouter_api_key,
            model=req.model,
            temperature=0,
            streaming=True,
        )
        self.req = req

    def _build_messages(self) -> list:
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        for turn in self.req.history:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            elif turn.role == "assistant":
                messages.append(AIMessage(content=turn.content))

        messages.append(HumanMessage(content=self.req.message))
        return messages

    async def stream(self) -> AsyncIterator[str]:
        messages = self._build_messages()
        try:
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as exc:
            yield json.dumps({"status": "error", "detail": str(exc)})