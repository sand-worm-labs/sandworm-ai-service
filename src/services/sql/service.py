from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from src.providers.openrouter import make_llm

from src.web.routes.sql.models import EditSqlRequest, FixSqlRequest
from .prompts import EDIT_SYSTEM_PROMPT, FIX_SYSTEM_PROMPT


class SqlService:
    def __init__(self, api_key: str, model: str) -> None:
        self.llm = make_llm(api_key=api_key, model=model)

    async def edit(self, req: EditSqlRequest) -> str:
        res = await self.llm.ainvoke([
            SystemMessage(content=EDIT_SYSTEM_PROMPT),
            HumanMessage(content=req.prompt),
        ])
        return res.content.strip()

    async def fix(self, req: FixSqlRequest) -> str:
        res = await self.llm.ainvoke([
            SystemMessage(content=FIX_SYSTEM_PROMPT),
            HumanMessage(content=req.error_message),
        ])
        return res.content.strip()
