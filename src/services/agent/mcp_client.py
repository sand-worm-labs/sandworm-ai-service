from __future__ import annotations

import json
from typing import Any

import httpx

# The MCP server is stateless (a fresh server per request), so each call is an
# independent JSON-RPC POST: no session to open or keep alive.
SKIP_TOOL_LOG_HEADER = "x-sandworm-skip-tool-log"


class McpError(Exception):
    pass


def _parse_body(response: httpx.Response) -> dict[str, Any]:
    """The server answers with JSON or a one-event SSE stream."""
    if "text/event-stream" in response.headers.get("content-type", ""):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        raise McpError("MCP response had no data event")
    return response.json()


class McpClient:
    def __init__(self, url: str, token: str, http: httpx.AsyncClient | None = None):
        self._url = url
        self._http = http or httpx.AsyncClient(timeout=120)
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        self._next_id = 0

    async def _rpc(self, method: str, params: dict[str, Any] | None = None, log: bool = False) -> dict[str, Any]:
        self._next_id += 1
        # The chat agent streams its own events for its chat, so the server must
        # not save the same calls a second time. A call made with `log` is
        # saved by the server instead: its prompt, then the work, in order.
        headers = self._headers if log else {**self._headers, SKIP_TOOL_LOG_HEADER: "1"}
        response = await self._http.post(
            self._url,
            headers=headers,
            json={"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}},
        )
        if response.status_code == 401:
            raise McpError("MCP rejected the user's token (expired or invalid)")
        response.raise_for_status()
        body = _parse_body(response)
        if "error" in body:
            raise McpError(body["error"].get("message", "MCP error"))
        return body["result"]

    async def instructions(self) -> str:
        result = await self._rpc(
            "initialize",
            {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "sandworm-ai", "version": "0"}},
        )
        return result.get("instructions", "")

    async def list_tools(self) -> list[dict[str, Any]]:
        return (await self._rpc("tools/list"))["tools"]

    async def call_tool(self, name: str, arguments: dict[str, Any], log: bool = False) -> tuple[str, bool]:
        """Returns (text, is_error). Tool failures come back as text for the model to read."""
        result = await self._rpc("tools/call", {"name": name, "arguments": arguments}, log)
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        return text, bool(result.get("isError"))

    async def aclose(self) -> None:
        await self._http.aclose()
