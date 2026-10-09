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
EDIT_NOTEBOOK = "edit_notebook"


def no_tool_call(tool: str, what: str) -> str:
    return f"You must apply the edit by calling {tool} with {what}."


NO_TOOL_CALL = no_tool_call(UPDATE_CELL, "the complete new content")


class CellEditFailed(Exception):
    pass


@dataclass
class CellRef:
    workspace_id: str
    document_id: str
    cell_id: str


def _new_value(call: dict[str, Any], key: str, transform: Callable[[str], str] | None) -> str | None:
    """The text the model chose, which is the only thing it decides."""
    try:
        value = json.loads(call["function"].get("arguments") or "{}").get(key)
    except json.JSONDecodeError:
        return None
    return transform(value) if transform and isinstance(value, str) else value


async def _apply(
    mcp: McpClient, tool: str, schema: dict[str, Any], fixed: dict[str, Any], key: str, value: str | None
) -> str | None:
    """Write the value with the tool. Returns what went wrong, or None when it worked.

    The ids always come from the request, never from the model.
    """
    arguments = {**fixed, key: value}
    if problem := invalid_arguments(schema, arguments):
        return problem
    try:
        text, is_error = await mcp.call_tool(tool, arguments, log=True)
    except (McpError, httpx.HTTPError) as exc:
        raise CellEditFailed(str(exc)) from exc
    return text if is_error else None


async def edit_with_mcp(
    mcp: McpClient,
    api_key: str,
    model: str,
    system: str,
    user: str,
    *,
    tool_name: str,
    fixed: dict[str, Any],
    key: str,
    what: str,
    transform: Callable[[str], str] | None = None,
) -> str:
    """Have the model write one value and apply it through the named MCP tool.

    `fixed` holds the ids the tool needs, taken from the request. `key` is the
    argument the model fills in. `transform` fixes up the text before it is
    written. Returns the text that was written. A call the schema or the server
    rejects goes back to the model for another try, up to MAX_ATTEMPTS.
    """
    server_tools = [t for t in await mcp.list_tools() if t["name"] == tool_name]
    if not server_tools:
        raise CellEditFailed(f"The notebook server has no {tool_name} tool.")
    tool = to_openrouter_tool(server_tools[0])
    forced = {"type": "function", "function": {"name": tool_name}}

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    problem = no_tool_call(tool_name, what)
    async with httpx.AsyncClient(timeout=120) as http:
        for _ in range(MAX_ATTEMPTS):
            message, _tokens = await call_model(http, api_key, model, messages, [tool], tool_choice=forced)
            calls = message.get("tool_calls")
            if not calls:
                problem = no_tool_call(tool_name, what)
                messages += [message, {"role": "user", "content": problem}]
                continue

            call = calls[0]
            messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": [call]})
            value = _new_value(call, key, transform)
            problem = await _apply(mcp, tool_name, tool["function"]["parameters"], fixed, key, value)
            if problem is None:
                return value
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": problem})

    raise CellEditFailed(problem)


async def update_cell_with_mcp(
    mcp: McpClient,
    api_key: str,
    model: str,
    system: str,
    user: str,
    cell: CellRef,
    transform: Callable[[str], str] | None = None,
    request: str | None = None,
) -> str:
    """Have the model write a cell's new text, and apply it through the MCP server's update_cell."""
    return await edit_with_mcp(
        mcp,
        api_key,
        model,
        system,
        user,
        tool_name=UPDATE_CELL,
        fixed={
            "notebookId": cell.document_id,
            "workspaceId": cell.workspace_id,
            "cellId": cell.cell_id,
            **({"request": request} if request else {}),
        },
        key="content",
        what="the complete new content",
        transform=transform,
    )


async def rename_notebook_with_mcp(
    mcp: McpClient,
    api_key: str,
    model: str,
    system: str,
    user: str,
    workspace_id: str,
    document_id: str,
    request: str | None = None,
) -> str:
    """Have the model write a notebook's new title, and apply it through the MCP server's edit_notebook."""
    return await edit_with_mcp(
        mcp,
        api_key,
        model,
        system,
        user,
        tool_name=EDIT_NOTEBOOK,
        fixed={"notebookId": document_id, "workspaceId": workspace_id, **({"request": request} if request else {})},
        key="title",
        what="the new title",
    )
