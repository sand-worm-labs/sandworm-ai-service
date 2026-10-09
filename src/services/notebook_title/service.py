from __future__ import annotations

import json

from src.config.settings import settings
from src.models.base import DocumentContext
from src.services.agent.cell_edit import rename_notebook_with_mcp
from src.services.agent.mcp_client import McpClient

MAX_CONTENT_CHARS = 6000
SYSTEM_PROMPT = (
    "You title data notebooks. Write a title of 3 to 8 words based on what the notebook actually does, with no "
    "quotes and no trailing punctuation, and avoid generic words like 'Notebook' or 'Analysis' on their own. "
    "The user wants a different title from the current one, so do not repeat it. "
    "Apply the title by calling edit_notebook with `title` set to it."
)


class NotebookTitleService:
    """Rename a notebook with AI, by having the MCP server change it.

    The notebook is read and renamed through the MCP server as the signed-in
    user, so nothing is returned for anyone else to write.
    """

    def __init__(self, api_key: str, model: str, context: DocumentContext) -> None:
        self.api_key = api_key
        self.model = model
        self.context = context

    async def rename(self) -> str:
        if not self.context.user_token:
            raise ValueError("Renaming a notebook needs the signed-in user's token to reach the notebook tools.")
        mcp = McpClient(settings.MCP_URL, self.context.user_token)
        try:
            current, content = await self._read(mcp)
            user = f"Current title: {current or '(none)'}\n\nNotebook content:\n{content or '(empty)'}"
            title = await rename_notebook_with_mcp(
                mcp,
                self.api_key,
                self.model,
                SYSTEM_PROMPT,
                user,
                self.context.workspace_id,
                self.context.document_id,
            )
            return title
        finally:
            await mcp.aclose()

    async def _read(self, mcp: McpClient) -> tuple[str, str]:
        """The current title and the start of the content. A notebook that cannot be read still gets a title."""
        try:
            text, is_error = await mcp.call_tool(
                "get_notebook", {"notebookId": self.context.document_id, "workspaceId": self.context.workspace_id}
            )
            if is_error:
                return "", ""
            notebook = json.loads(text)
            return str(notebook.get("title") or ""), str(notebook.get("content") or "")[:MAX_CONTENT_CHARS]
        except Exception:
            return "", ""
