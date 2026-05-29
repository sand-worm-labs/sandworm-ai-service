from __future__ import annotations

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

from src.services.completions.models import CompletionRequest
from src.web.routes.document.models import DocumentTitleRequest


CHAT_TITLE_PROMPT = """Generate a short, concise title for this conversation.
3-6 words max. No punctuation. No quotes. Just the title."""

DOCUMENT_TITLE_PROMPT = """Generate a short, concise title for this analytics document.
3-6 words max. No punctuation. No quotes. Just the title."""


class TitleService:
    def __init__(self, req: CompletionRequest) -> None:
        self.llm = ChatOpenRouter(
            api_key=req.api_key,
            model=req.model,
            temperature=0,
        )
        self.req = req

    async def generate(self) -> str:
        first_user = next((m for m in self.req.messages if m.role == "user"), None)
        if not first_user:
            return "New Chat"

        messages = [
            SystemMessage(content=CHAT_TITLE_PROMPT),
            HumanMessage(content=first_user.content),
        ]

        if self.req.derived_context:
            messages = [SystemMessage(content=self.req.derived_context), *messages]

        res = await self.llm.ainvoke(messages)
        return res.content.strip()


class DocumentTitleService:
    def __init__(self, req: DocumentTitleRequest) -> None:
        self.llm = ChatOpenRouter(
            api_key=req.openrouter_api_key,
            model=req.model,
            temperature=0,
        )
        self.req = req

    async def generate(self) -> str:
        res = await self.llm.ainvoke([
            SystemMessage(content=DOCUMENT_TITLE_PROMPT),
            HumanMessage(content=self.req.message),
        ])
        return res.content.strip()