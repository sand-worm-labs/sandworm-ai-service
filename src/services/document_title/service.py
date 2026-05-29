from __future__ import annotations

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

from src.web.routes.document.models import DocumentTitleRequest


SYSTEM_PROMPT = """Generate a short, concise title for this analytics document.
3-6 words max. No punctuation. No quotes. Just the title."""


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
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self.req.message),
        ])
        return res.content.strip()
