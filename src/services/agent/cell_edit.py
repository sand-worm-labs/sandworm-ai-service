from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from src.services.agent.mcp_client import McpClient, McpError
from src.services.agent.model import call_model
from src.services.agent.tools import invalid_arguments, to_openrouter_tool

UPDATE_CELL = "update_cell"
MAX_ATTEMPTS = 2


class CellEditFailed(Exception):
    pass


@dataclass
class CellRef:
    workspace_id: str
    document_id: str
    cell_id: str


async def update_cell_with_mcp(
    mcp: McpClient,
    api_key: str,
    model: str,
    system: str,
    user: str,
    cell: CellRef,
    transform: Callable[[str], str] | None = None,
) -> str:
    """Have the model write a cell's new text, and apply it through the MCP server's update_cell.

    The model decides only the content: the ids always come from the request,
    never from the model. `transform` fixes up the text before it is written.
    Returns the text that was written. A call the schema or the server rejects
    goes back to the model once to correct.
    """
    server_tools = [t for t in await mcp.list_tools() if t["name"] == UPDATE_CELL]
    if not server_tools:
        raise CellEditFailed(f"The notebook server has no {UPDATE_CELL} tool.")
    tools = [to_openrouter_tool(server_tools[0])]
    schema = tools[0]["function"]["parameters"]
    forced = {"type": "function", "function": {"name": UPDATE_CELL}}

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    problem = "The model did not call update_cell."
    async with httpx.AsyncClient(timeout=120) as http:
        for _ in range(MAX_ATTEMPTS):
            message, _tokens = await call_model(http, api_key, model, messages, tools, tool_choice=forced)
            calls = message.get("tool_calls") or []
            if not calls:
                problem = "You must apply the edit by calling update_cell with the complete new content."
                messages += [message, {"role": "user", "content": problem}]
                continue

            call = calls[0]
            messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": [call]})
            try:
                content = json.loads(call["function"].get("arguments") or "{}").get("content")
            except json.JSONDecodeError:
                content = None
            if transform and isinstance(content, str):
                content = transform(content)
            arguments = {
                "notebookId": cell.document_id,
                "workspaceId": cell.workspace_id,
                "cellId": cell.cell_id,
                "content": content,
            }
            problem = invalid_arguments(schema, arguments)
            if not problem:
                try:
                    text, is_error = await mcp.call_tool(UPDATE_CELL, arguments)
                except (McpError, httpx.HTTPError) as exc:
                    raise CellEditFailed(str(exc)) from exc
                if not is_error:
                    return content
                problem = text
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": problem})

    raise CellEditFailed(problem)
