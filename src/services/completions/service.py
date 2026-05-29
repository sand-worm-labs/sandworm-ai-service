from __future__ import annotations

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .models import CompletionInput, CompletionOutput

class CompletionService:
    async def complete(self, input: CompletionInput) -> CompletionOutput:
        llm = ChatOpenRouter(api_key=input.api_key, model=input.model, temperature=0)

        messages = [
            HumanMessage(content=m.content) if m.role == "user"
            else AIMessage(content=m.content) if m.role == "assistant"
            else SystemMessage(content=m.content)
            for m in input.messages
        ]

        res = await llm.ainvoke(messages)
        return CompletionOutput(content=res.content, finish_reason=res.response_metadata.get("finish_reason", "stop"))