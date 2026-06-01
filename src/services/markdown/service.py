from __future__ import annotations

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

from src.web.routes.markdown.models import EditMarkdownRequest


EDIT_SYSTEM_PROMPT = """You are an expert technical writer.
Edit the provided markdown content according to the user's instructions.
Return only the raw markdown with no extra commentary."""


class MarkdownService:
    def __init__(self, api_key: str, model: str) -> None:
        self.llm = ChatOpenRouter(api_key=api_key, model=model, temperature=0)

    async def edit(self, req: EditMarkdownRequest) -> str:
        res = await self.llm.ainvoke([
            SystemMessage(content=EDIT_SYSTEM_PROMPT),
            HumanMessage(content=req.prompt),
        ])
        return res.content.strip()
