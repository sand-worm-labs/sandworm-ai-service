from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from src.providers.openrouter import make_llm

from src.services.completions.models import CompletionRequest
from .prompts import SYSTEM_PROMPT


class ChatTitleService:
    def __init__(self, req: CompletionRequest) -> None:
        self.llm = make_llm(
            api_key=req.openrouter_api_key,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        self.req = req

    async def generate(self) -> str:
        first_user = next((m for m in self.req.messages if m.role == "user"), None)
        if not first_user:
            return "New Chat"

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=first_user.content),
        ]

        if self.req.derived_context:
            messages = [SystemMessage(content=self.req.derived_context), *messages]

        res = await self.llm.ainvoke(messages)
        return res.content.strip()
