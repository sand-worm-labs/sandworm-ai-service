from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from src.providers.openrouter import make_llm

from src.web.routes.code.models import EditCodeRequest, FixCodeRequest


EDIT_SYSTEM_PROMPT = """You are an expert Python data scientist.
Edit the provided code according to the user's instructions.
Return only the raw Python code. Do NOT wrap the output in ```python or any other code fences. No explanations, no preamble."""

FIX_SYSTEM_PROMPT = """You are an expert Python data scientist.
Fix the code based on the error message provided.
Return only the corrected raw Python code. Do NOT wrap the output in ```python or any other code fences. No explanations, no preamble."""


class CodeService:
    def __init__(self, api_key: str, model: str) -> None:
        self.llm = ChatOpenRouter(api_key=api_key, model=model, temperature=0)

    async def edit(self, req: EditCodeRequest) -> str:
        res = await self.llm.ainvoke([
            SystemMessage(content=EDIT_SYSTEM_PROMPT),
            HumanMessage(content=req.prompt),
        ])
        return res.content.strip()

    async def fix(self, req: FixCodeRequest) -> str:
        res = await self.llm.ainvoke([
            SystemMessage(content=FIX_SYSTEM_PROMPT),
            HumanMessage(content=req.error_message),
        ])
        return res.content.strip()
