from __future__ import annotations

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .models import CompletionInput, CompletionOutput


class CompletionService:
    async def stream(self, input: CompletionInput) -> AsyncIterator[str]:
        llm = ChatOpenRouter(api_key=input.api_key, model=input.model, temperature=0, streaming=True)

        messages = [
            HumanMessage(content=m.content) if m.role == "user"
            else AIMessage(content=m.content) if m.role == "assistant"
            else SystemMessage(content=m.content)
            for m in input.messages
        ]

        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content