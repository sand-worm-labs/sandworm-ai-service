from __future__ import annotations

import uuid
from typing import Literal

from src.config.settings import settings
from src.models.base import DocumentContext
from src.services.agent.cell_edit import CellEditFailed, CellRef, update_cell_with_mcp
from src.services.agent.mcp_client import McpClient
from src.services.prompt_rules import PYTHON_TABLE_RULE
from src.services.research_memory.models import Attempt
from src.services.research_memory.service import recall_text, remember_in_background
from src.services.table_output import render_tables

CellKind = Literal["python", "sql", "markdown"]

# Editing a cell is the same job for every cell type: only the persona and
# the output format change.
_PERSONA = {
    "python": "an expert Python data scientist",
    "sql": "an expert SQL data analyst",
    "markdown": "an expert technical writer",
}
_NOUN = {"python": "code", "sql": "SQL query", "markdown": "markdown content"}
_RAW = {"python": "Python code", "sql": "SQL", "markdown": "markdown"}
_FENCE = {"python": "```python", "sql": "```sql", "markdown": "```markdown"}
_TAIL = {
    "python": "No explanations, no preamble.",
    "sql": "No explanations, no preamble.",
    "markdown": "No extra commentary, no preamble.",
}

MEMORY_NOTE = (
    "\n\nWhat this workspace already learned and tried is below. Use it to avoid approaches that failed before.\n"
)


def _apply(kind: CellKind, task: str) -> str:
    # Python cells render a DataFrame as a table, so tables must not be printed.
    rules = f"{PYTHON_TABLE_RULE}\n" if kind == "python" else ""
    return (
        f"{task}\n"
        f"{rules}"
        f"Apply the result by calling update_cell with `content` set to the COMPLETE new {_NOUN[kind]}: "
        f"the raw {_RAW[kind]} only, never wrapped in {_FENCE[kind]} or any other code fences. {_TAIL[kind]}"
    )


def edit_prompt(kind: CellKind) -> str:
    return _apply(kind, f"You are {_PERSONA[kind]}.\nEdit the provided {_NOUN[kind]} according to the user's instructions.")


def fix_prompt(kind: Literal["python", "sql"]) -> str:
    return _apply(kind, f"You are {_PERSONA[kind]}.\nFix the {_NOUN[kind]} based on the error message provided.")


class CellService:
    """Edit or fix one cell with AI, by having the MCP server change it.

    The model writes the new text and the MCP server's update_cell applies it,
    so nothing here returns text for anyone else to write. Notebook memory
    applies as in chat: what the workspace already learned and tried goes into
    the prompt, and each edit or fix is saved as an attempt, now with a known
    outcome. This service never reads the notebook directly.
    """

    def __init__(self, api_key: str, model: str, kind: CellKind, context: DocumentContext, block_id: str) -> None:
        self.api_key = api_key
        self.model = model
        self.kind = kind
        self.context = context
        self.block_id = block_id

    async def _apply_edit(self, action: str, system: str, user: str) -> None:
        if not self.context.user_token:
            raise ValueError("Editing a cell needs the signed-in user's token to reach the notebook tools.")
        memory = await recall_text(self.context.workspace_id, user, self.context.document_id)
        cell = CellRef(self.context.workspace_id, self.context.document_id, self.block_id)
        mcp = McpClient(settings.MCP_URL, self.context.user_token)
        try:
            written = await update_cell_with_mcp(
                mcp,
                self.api_key,
                self.model,
                system + (MEMORY_NOTE + memory if memory else ""),
                user,
                cell,
                # Rewrites a printed table into one the notebook can render, so the rule
                # does not depend on the model following it.
                transform=render_tables if self.kind == "python" else None,
            )
            self._remember(action, user, True, f"wrote: {written}")
        except CellEditFailed as exc:
            self._remember(action, user, False, str(exc))
            raise
        finally:
            await mcp.aclose()

    def _remember(self, action: str, request: str, worked: bool, result: str) -> None:
        remember_in_background(
            self.context.workspace_id,
            self.context.document_id,
            f"{action}-{uuid.uuid4()}",
            Attempt(
                f"{action}_{self.kind}",
                f"request={' '.join(request.split())[:300]}",
                worked,
                " ".join(result.split())[:400],
            ),
        )

    async def edit(self, prompt: str) -> None:
        await self._apply_edit("edit", edit_prompt(self.kind), prompt)

    async def fix(self, error_message: str) -> None:
        if self.kind == "markdown":
            raise ValueError("Markdown cells have nothing to fix")
        await self._apply_edit("fix", fix_prompt(self.kind), error_message)
