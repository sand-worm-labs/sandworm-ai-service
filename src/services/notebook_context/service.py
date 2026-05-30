from __future__ import annotations

import httpx

from src.services.intent.models import IntentClass
from .models import AiContextRequest, NotebookContextMode


def resolve_notebook_mode(
    intent_class: IntentClass,
    has_focused_blocks: bool,
    has_non_explicit_focus: bool,
) -> NotebookContextMode:
    if intent_class == IntentClass.CONVERSATIONAL:
        return "none"

    if intent_class == IntentClass.EXPLANATORY:
        if has_focused_blocks or has_non_explicit_focus:
            return "full"
        return "none"

    if intent_class == IntentClass.EDITORIAL:
        return "full"

    if intent_class == IntentClass.ANALYTICAL:
        if has_focused_blocks or has_non_explicit_focus:
            return "full"
        return "spine"


class NotebookContextService:
    def __init__(
        self,
        nest_base_url: str,
        intent: IntentClass,
        has_focused_blocks: bool = False,
        has_non_explicit_focus: bool = False,
    ):
        self.nest_base_url = nest_base_url.rstrip("/")
        self.intent = intent
        self.has_focused_blocks = has_focused_blocks
        self.has_non_explicit_focus = has_non_explicit_focus

    async def fetch(self, ctx: AiContextRequest) -> str | None:
        mode = resolve_notebook_mode(
            self.intent,
            self.has_focused_blocks,
            self.has_non_explicit_focus,
        )

        if mode == "none":
            return None

        params: dict = {
            "workspaceId": ctx.workspace_id,
            "documentId": ctx.document_id,
            "mode": mode,
        }

        if ctx.focused_block_ids:
            params["focusedBlockIds"] = ",".join(ctx.focused_block_ids)

        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(
                    f"{self.nest_base_url}/yjs_documents/{ctx.document_id}/ai-context",
                    params=params,
                )
                res.raise_for_status()
                return res.text
        except httpx.ConnectError:
            return None