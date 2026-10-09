from __future__ import annotations

from typing import Any

from jsonschema import Draft7Validator

from src.config.settings import settings


def to_openrouter_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """MCP tool definition -> OpenAI/OpenRouter function tool."""
    schema = {k: v for k, v in (tool.get("inputSchema") or {}).items() if k != "$schema"}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return {
        "type": "function",
        "function": {"name": tool["name"], "description": tool.get("description", ""), "parameters": schema},
    }


def _names(raw: str) -> set[str]:
    return {n.strip() for n in raw.split(",") if n.strip()}


def select_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply AGENT_INCLUDE_TOOLS / AGENT_EXCLUDE_TOOLS to the MCP tool list."""
    include, exclude = _names(settings.AGENT_INCLUDE_TOOLS), _names(settings.AGENT_EXCLUDE_TOOLS)
    return [t for t in tools if (not include or t["name"] in include) and t["name"] not in exclude]


def invalid_arguments(schema: dict[str, Any], arguments: Any) -> str | None:
    """Check a tool call against its input schema. Returns what is wrong, or None."""
    problems = sorted(Draft7Validator(schema).iter_errors(arguments), key=lambda e: list(e.path))
    if not problems:
        return None
    lines = [f"{'.'.join(map(str, e.path)) or 'arguments'}: {e.message}" for e in problems[:5]]
    return "Invalid arguments, nothing was run. Fix these and call the tool again:\n- " + "\n- ".join(lines)
