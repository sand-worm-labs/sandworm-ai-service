from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Public data APIs a python block can fetch from with plain `requests`. A
# notebook is built from these alone when the user asks for open data, or when
# chain SQL (Dune) cannot run: see use_open_data below.
#
# sources.json is the same catalog apps/mcp serves to agents
# (apps/mcp/src/tools/notebooks/open-data.ts), exported as JSON. That file is
# the one to edit; regenerate this copy from apps/mcp with:
#   node --input-type=module -e "import { OPEN_DATA_SOURCES as s } from './src/tools/notebooks/open-data.ts'; \
#     import { writeFileSync as w } from 'node:fs'; \
#     w('../ai/src/services/open_data/sources.json', JSON.stringify(s, null, 2) + '\n')"
OPEN_DATA_SOURCES: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "sources.json").read_text(encoding="utf-8")
)

MATCHES_PER_QUERY = 3
# With nothing matched, these two still answer most market questions.
FALLBACK_IDS = ("defillama", "coingecko")

_STOPWORDS = {
    "the", "and", "for", "with", "over", "last", "show", "all", "per", "how", "what", "are", "from", "that",
    "this", "into", "its", "our", "you", "your", "can", "has", "have", "was", "were", "will", "who", "which",
    "when", "where",
}


# Same wording apps/mcp reads a prompt for (tools/notebooks/data-mode.ts).
_OPEN = re.compile(r"\bopen[- ]?data\b|\b(public|open|free) apis?\b|\bpublic data\b", re.IGNORECASE)
_SANDWORM = re.compile(
    r"\bsandworm('s)? (data|tools?|cloud)\b|\bdune\b|\bpower ?tools?\b"
    r"|\b(our|local|own|internal) (data|tools?)\b|\bchain data\b",
    re.IGNORECASE,
)


# What a prompt asks for: True for open data, False for Sandworm's own data and
# tools, None when it names neither or both.
def open_data_from_prompt(prompt: str) -> bool | None:
    asks_open = bool(_OPEN.search(prompt))
    asks_sandworm = bool(_SANDWORM.search(prompt))
    return None if asks_open == asks_sandworm else asks_open


# Without chain SQL (a free workspace, or no source connected) it is always
# open data. Otherwise the latest prompt that says which data to use decides,
# and with none it is Sandworm's data.
def use_open_data(prompts: list[str], sql_available: bool) -> bool:
    if not sql_available:
        return True
    for prompt in reversed(prompts):
        asked = open_data_from_prompt(prompt)
        if asked is not None:
            return asked
    return False


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def tokens(text: str) -> list[str]:
    words = re.split(r"[^a-z0-9]+", text.lower())
    return [
        _singular(w)
        for w in words
        if (len(w) > 2 or re.search(r"\d", w)) and w not in _STOPWORDS
    ]


# Keyword ranking, same as apps/mcp: a word counts most in keywords and the name.
def _score(source: dict[str, Any], words: list[str]) -> int:
    strong = {*source["keywords"], *tokens(source["name"])}
    weak = set(tokens(source["covers"]))
    return sum(3 if w in strong else 1 if w in weak else 0 for w in words)


def match_open_data(query: str, limit: int = MATCHES_PER_QUERY) -> list[dict[str, Any]]:
    words = tokens(query)
    scored = [(source, _score(source, words)) for source in OPEN_DATA_SOURCES]
    ranked = [source for source, score in sorted(scored, key=lambda m: -m[1]) if score > 0]
    if not ranked:
        ranked = [s for s in OPEN_DATA_SOURCES if s["id"] in FALLBACK_IDS]
    return ranked[:limit]


def _key_line(source: dict[str, Any]) -> str:
    key = source.get("key")
    if not key:
        return "none needed"
    need = "required" if key["required"] else "optional"
    return f"os.environ[\"{key['env']}\"] ({need}; send as {key['send']})"


def describe_source(source: dict[str, Any], with_endpoints: bool = True) -> str:
    lines = [f"- {source['name']}: {source['covers']}. Base URL: {source['baseUrl']}. API key: {_key_line(source)}"]
    if source.get("limits"):
        lines.append(f"  Limits: {source['limits']}")
    if source.get("notes"):
        lines.append(f"  Note: {source['notes']}")
    if with_endpoints:
        lines.extend(f"  {e['path']} -> {e['gives']}" for e in source["endpoints"])
    return "\n".join(lines)


# Appended to the block planner's input: which public APIs cover each sub-goal,
# and how the plan changes while chain data is offline.
def planner_context(queries: list[str]) -> str:
    seen: dict[str, dict[str, Any]] = {}
    for query in queries:
        for source in match_open_data(query):
            seen.setdefault(source["id"], source)

    sources = "\n".join(describe_source(s, with_endpoints=False) for s in seen.values())
    return (
        "**Build this from open data only.** Dune and the power tools are not used, so this "
        "overrides the rules above: do NOT plan a sql block that pulls from Dune, and do "
        "NOT plan a power_toolbox block. The first block for every sub-goal is a python "
        "block that fetches from one of the public APIs below with an HTTP request and "
        "leaves the result as a DataFrame. Every other block type works on top of it "
        "exactly as on top of a sql block: a visualization for each chart, a pivot_table "
        "for a tabular summary, a sql block that depends_on it for a local follow-up "
        "query, plus markdown, rich_text, dashboard_header and input blocks as usual. "
        "Combine APIs when one does not cover a sub-goal, and skip a sub-goal none of "
        "them can answer. Name the API in each python block's description.\n\n"
        f"**Public APIs that fit these sub-goals:**\n{sources}"
    )


OPEN_DATA_PYTHON_RULES = (
    " This notebook uses open data only, not Dune: get the data with HTTP requests to the public APIs "
    "listed in the task, using the `requests` library with timeout=60, retrying a "
    "dropped connection or a 429 once or twice. Read any API key from os.environ; "
    "never write a key into the code. Build a pandas DataFrame from the response. "
    "Leave charting to the visualization blocks that follow, unless the task itself "
    "asks for a chart here. Never invent data: if a request fails, let the error raise."
)


# Appended to a python block's task: the endpoints of the APIs that fit it.
def block_context(task: str) -> str:
    sources = "\n".join(describe_source(s) for s in match_open_data(task))
    return (
        "**Public APIs to fetch from** (a path starting with / is relative to the "
        f"source's base URL; pick the endpoints this task needs):\n{sources}"
    )


# On the free plan only public APIs can be used. These notes make the intent
# step mark chain-only parts as not feasible, and the reply tell the user why.
PAID_PLAN_REASON = "needs Sandworm chain data (paid plan)"

FREE_PLAN_INTENT_NOTE = (
    "This workspace is on the free plan: Sandworm's chain data (Dune and Sandworm Cloud SQL) "
    "is not available, only free public APIs. A sub_goal that needs chain data no free public "
    "API exposes, such as decoded contract events, every transaction or transfer of a contract, "
    "wallet-level histories or holder lists, is feasible=false with reason "
    f'"{PAID_PLAN_REASON}". Never stand in a different public metric for it.'
)


def free_plan_reply_note(left_out: list[str]) -> str:
    parts = "; ".join(left_out)
    return (
        "This workspace is on the free plan. These parts were left out because they need "
        f"Sandworm's chain data, which comes with a paid plan: {parts}. Say this plainly in "
        "your reply and tell the user they can upgrade in Settings > Plan to get them. Do not "
        "offer a workaround that pretends to answer them."
    )
