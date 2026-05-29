from __future__ import annotations
import httpx

from .models import AiContextRequest

class NotebookContextService:
    def __init__(self, nest_base_url: str):
        self.nest_base_url = nest_base_url.rstrip("/")

    async def fetch(self, ctx: AiContextRequest) -> str | None:
        params: dict = {"workspaceId": ctx.workspace_id, "documentId": ctx.document_id}
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