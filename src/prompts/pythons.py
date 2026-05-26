# ══════════════════════════════════════════════════════════════════════
# ⬢  BLOCK — PYTHON
#    System directives and user message builders for all Python actions.
#
#    Actions covered:
#      python_edit     — modify existing code per user instructions
#      python_fix      — repair code that produced a runtime error
#      python_generate — write code from scratch given intent + scope
#
#    Execution environment assumptions (keep in sync with NestJS executor):
#      import pandas as pd
#      import plotly.express as px
#      import plotly.graph_objects as go
#      import duckdb
#      from web3 import Web3
# ══════════════════════════════════════════════════════════════════════

from ..types import PromptContext

_ENV_NOTE = """\
The execution environment provides: `pandas` as `pd`, `plotly.express` as `px`, \
`plotly.graph_objects` as `go`, `duckdb`, `web3.Web3`. Do not re-import these. \
Do not install packages.\
"""


def build_directive_edit(ctx: PromptContext) -> str:
    return f"""\
## Task: Edit Python Code

Apply the user's instructions to the existing code. Return **only** the \
modified Python — no explanation, no markdown fences, no preamble.

Rules:
- Preserve imports, variable names, and overall structure where the instruction allows
- Make the minimal diff needed — do not refactor what wasn't asked
- {_ENV_NOTE}
- If the instruction implies a new dependency that isn't in the environment, \
use the closest available alternative\
"""


def build_directive_fix(ctx: PromptContext) -> str:
    return f"""\
## Task: Fix Python Code

The code raised an exception. Diagnose and fix it. Return **only** the \
corrected Python — no explanation, no markdown fences, no preamble.

Rules:
- Fix only what caused the error — do not restructure unrelated code
- {_ENV_NOTE}
- For `KeyError` / `AttributeError`: verify column/attribute names against scope
- For `TypeError` on numeric ops: check if column is object dtype, add explicit cast
- For `MemoryError`: add chunking or sampling — do not silently drop data \
without a comment explaining it\
"""

def build_directive_generate(ctx: PromptContext) -> str:
    return f"""\
## Task: Generate Python Code

Write Python code that satisfies the analytical intent. Return **only** the \
code — no explanation, no markdown fences, no preamble.

Rules:
- {_ENV_NOTE}
- Reference dataframes and variables from scope directly — do not redeclare them
- For blockchain data: normalise token amounts (`/ 10**decimals`), \
handle hex addresses as lowercase strings, format wei as ETH where displayed
- For visualizations: use plotly (not matplotlib), assign the figure \
to a variable named `fig` — the runtime will display it automatically
- Add a brief inline comment on any non-obvious step\
"""



def build_user_edit(ctx: PromptContext) -> str:
    return "\n\n".join(filter(None, [
        f"**Current code:**\n```python\n{ctx.source.strip()}\n```",
        f"**Instructions:** {ctx.instructions.strip()}",
    ]))


def build_user_fix(ctx: PromptContext) -> str:
    parts = [
        f"**Current code:**\n```python\n{ctx.source.strip()}\n```",
        f"**Error:**\n```\n{ctx.error_message.strip()}\n```",
    ]
    if ctx.instructions:
        parts.append(f"**Additional context:** {ctx.instructions.strip()}")
    parts.append("Fix the error. Return only the corrected Python.")
    return "\n\n".join(parts)


def build_user_generate(ctx: PromptContext) -> str:
    return "\n\n".join(filter(None, [
        f"**Intent:** {ctx.instructions.strip()}",
        (
            f"**Preceding block output (for reference):**\n{ctx.source.strip()}"
            if ctx.source else None
        ),
    ]))