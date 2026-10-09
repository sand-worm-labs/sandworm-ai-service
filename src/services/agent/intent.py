from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config.settings import settings
from src.models.base import Message
from src.services.agent.model import call_model

log = logging.getLogger("sandworm.agent")

# What each kind of message may use. "chat" gets no tools at all: the model just answers.
READ = ["get_notebook", "get_run_results", "get_dashboard"]
EDIT = ["get_notebook", "add_cell", "update_cell", "delete_cell", "edit_header", "list_data_sources", "get_data_source_schema", "search_tools"]
RUN = ["get_notebook", "run_notebook", "get_run_results"]

INTENT_TOOLS: dict[str, list[str] | None] = {
    "chat": [],
    "read": READ,
    "edit": EDIT,
    "run": RUN,
    "build": sorted({*EDIT, *RUN}),
    "dashboard": ["get_notebook", "get_dashboard", "set_dashboard", "edit_header"],
    # Not sure: offer everything the agent is allowed.
    "general": None,
}

DESCRIPTIONS = {
    "chat": "greetings, thanks, or a general question that needs no look at the notebook (what is a DEX, how does X work)",
    "read": "asks about what is in the notebook or its results, without changing anything (explain, summarize, what does cell 3 do)",
    "edit": "change, fix, add, remove or rewrite specific cells or text in the notebook",
    "run": "run the notebook or some cells, or check on a run",
    "build": "do new analysis: write queries or code, run them and show results",
    "dashboard": "change what is shown on the dashboard, or its heading",
}

SYSTEM = (
    "Classify the user's last message for a notebook assistant. Reply with exactly one word, one of:\n"
    + "\n".join(f"- {name}: {text}" for name, text in DESCRIPTIONS.items())
    + "\n- general: none of these clearly fits\nWhen unsure between chat and anything else, pick the other one."
)


def parse_intent(text: str) -> str:
    words = text.strip().lower().replace(".", " ").replace(":", " ").split()
    return words[0] if words and words[0] in INTENT_TOOLS else "general"


async def classify(http: httpx.AsyncClient, api_key: str, model: str, messages: list[Message]) -> str:
    """One cheap model call that labels the latest message. Any trouble means "general"."""
    if not settings.AGENT_INTENT_ENABLED or not messages:
        return "general"
    recent = [{"role": m.role, "content": m.content[:1500]} for m in messages[-4:]]
    try:
        message, _ = await call_model(
            http,
            api_key,
            settings.AGENT_INTENT_MODEL or model,
            [{"role": "system", "content": SYSTEM}, *recent],
            temperature=0.0,
            max_tokens=8,
        )
    except (httpx.HTTPError, RuntimeError, KeyError, IndexError) as exc:
        log.warning("intent classification failed, offering every allowed tool: %s", exc)
        return "general"
    return parse_intent(message.get("content") or "")


def narrow(tools: list[dict[str, Any]], intent: str) -> list[dict[str, Any]]:
    """Keep only the tools this intent may use out of the ones the agent is allowed."""
    names = INTENT_TOOLS.get(intent)
    if names is None:
        return tools
    return [t for t in tools if t["function"]["name"] in names]
