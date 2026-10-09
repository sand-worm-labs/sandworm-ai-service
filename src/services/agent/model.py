from __future__ import annotations

from typing import Any

import httpx

from src.config.settings import settings


async def call_model(
    http: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    tool_choice: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """The one place a model is called (OpenRouter, with the user's own key).

    Returns the assistant message and the tokens it used.
    """
    payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if tools:
        # One call at a time: notebook edits (add/update/delete cell) are order-dependent.
        payload |= {"tools": tools, "parallel_tool_calls": False}
        if tool_choice:
            payload["tool_choice"] = tool_choice

    response = await http.post(
        f"{settings.openrouter_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    )
    if tools and response.status_code == 404 and "tool" in response.text.lower():
        raise RuntimeError(f"The model {model} cannot call tools, which chat needs. Pick another model in the model menu.")
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"], int((body.get("usage") or {}).get("total_tokens") or 0)


async def ask(
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """A single question and answer, for jobs that only need text back (titles, cell edits and fixes)."""
    async with httpx.AsyncClient(timeout=120) as http:
        message, _ = await call_model(
            http,
            api_key,
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return (message.get("content") or "").strip()
