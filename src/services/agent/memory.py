from __future__ import annotations

import json
import logging
from typing import Any

from src.services.agent.mcp_client import McpClient
from src.services.research_memory.models import Attempt
from src.services.research_memory.service import recall_text
from src.services.research_memory.store import index_attempts, index_notebook

log = logging.getLogger("sandworm.agent")

# Reading tools say nothing about what the AI tried to do, so they are not remembered.
_READ_ONLY_PREFIXES = ("get_", "list_", "search_", "read_")
_ALWAYS_REMEMBER = {"get_run_results"}
RESULT_CHARS = 400
CONTENT_CHARS = 600


def attempt_for(name: str, arguments: dict[str, Any], result: str, is_error: bool) -> Attempt | None:
    """What to remember about one tool call, or None if it is not worth remembering."""
    if name.startswith(_READ_ONLY_PREFIXES) and name not in _ALWAYS_REMEMBER:
        return None
    details = [f"{k}={arguments[k]}" for k in ("type", "title", "dataSource", "dataframeName") if arguments.get(k)]
    if arguments.get("content"):
        details.append(f"content={' '.join(str(arguments['content']).split())[:CONTENT_CHARS]}")
    if not details:
        details.append(json.dumps(arguments, default=str)[:CONTENT_CHARS])
    return Attempt(name, " ".join(details), not is_error, " ".join(result.split())[:RESULT_CHARS])


async def prior_research(workspace_id: str, document_id: str, question: str) -> str:
    """What this workspace already learned that bears on the user's request."""
    return await recall_text(workspace_id, question, document_id)


async def remember(workspace_id: str, document_id: str, run_id: str, attempts: list[Attempt], mcp: McpClient) -> None:
    """Store the notebook and what the AI tried in it, for later chats. Read through the MCP like everything else."""
    title = "Untitled notebook"
    try:
        text, is_error = await mcp.call_tool("get_notebook", {"notebookId": document_id, "workspaceId": workspace_id})
        if not is_error:
            notebook = json.loads(text)
            title = notebook.get("title") or title
            await index_notebook(workspace_id, document_id, title, notebook.get("content") or "")
    except Exception:
        log.warning("could not remember notebook %s", document_id, exc_info=True)
    try:
        # Kept even when the notebook itself could not be read.
        await index_attempts(workspace_id, document_id, title, run_id, attempts)
    except Exception:
        log.warning("could not remember attempts for notebook %s", document_id, exc_info=True)
