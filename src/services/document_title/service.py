from __future__ import annotations

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

from src.config.settings import settings
from src.web.routes.document.models import DocumentTitleRequest
from src.services.notebook_context.service import NotebookContextService
from src.services.notebook_context.models import AiContextRequest
from src.services.intent.models import IntentClass


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
        notebook_markdown = await NotebookContextService(
            nest_base_url=settings.nest_base_url,
            intent=IntentClass.EDITORIAL,
        ).fetch(AiContextRequest(
            document_id=self.req.context.document_id,
            workspace_id=self.req.context.workspace_id,
        ))

        user_content = self.req.message
        if notebook_markdown:
            user_content = f"{self.req.message}\n\n<document>\n{notebook_markdown}\n</document>"

        res = await self.llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        return res.content.strip()
