from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.providers.openrouter import make_streaming_llm
from .models import CompletionRequest


class CompletionService:
    async def stream(self, input: CompletionRequest) -> AsyncIterator[str]:
        llm = make_streaming_llm(api_key=input.openrouter_api_key, model=input.model)

        messages = [
            HumanMessage(content=m.content) if m.role == "user"
            else AIMessage(content=m.content) if m.role == "assistant"
            else SystemMessage(content=m.content)
            for m in input.messages
        ]

        if input.derived_context:
            messages = [SystemMessage(content=input.derived_context), *messages]
            
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content