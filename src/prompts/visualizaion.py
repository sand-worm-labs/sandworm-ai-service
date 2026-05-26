# ══════════════════════════════════════════════════════════════════════
# ⬢  BLOCK — VISUALIZATION
#    Generates plotly visualization code from preceding query results.
#
#    The user's dataframe(s) come from BlockScope. The LLM produces
#    Python code that creates a plotly figure assigned to `fig`.
#    The Sandworm runtime detects `fig` and renders it automatically.
# ══════════════════════════════════════════════════════════════════════

from ..types import PromptContext


# ─── SYSTEM DIRECTIVE ─────────────────────────────────────────────────

def build_directive(ctx: PromptContext) -> str:
    return """\
## Task: Generate Visualization Code

Write Python code that creates a plotly visualization from the available \
dataframe(s). Return **only** the Python code — no explanation, \
no markdown fences, no preamble.

Rules:
- Use `plotly.express` (`px`) for standard charts; `plotly.graph_objects` (`go`) \
for custom/composite layouts
- Assign the final figure to a variable named **`fig`** — the runtime renders it
- Do not call `fig.show()` — the runtime handles display
- Reference dataframes from scope directly — do not re-query or re-fetch data
- For time series: x-axis should be the time column, use `DATE_TRUNC` granularity \
from the query where applicable
- For token amounts displayed to users: format as human-readable \
(e.g. `/ 1e18` for ETH, abbreviated with `K`/`M`/`B` suffixes)
- Apply a clean, minimal layout: `template='plotly_dark'` or `template='plotly_white'` \
matching the notebook theme — do not hardcode colors unless the user specifies them
- Add a descriptive title and axis labels — never leave them blank\
"""


# ─── USER MESSAGE BUILDER ─────────────────────────────────────────────

def build_user_generate(ctx: PromptContext) -> str:
    parts: list[str] = []

    if ctx.instructions:
        parts.append(f"**Visualization intent:** {ctx.instructions.strip()}")

    if ctx.source.strip():
        parts.append(
            f"**Preceding query / data shape:**\n```\n{ctx.source.strip()}\n```"
        )

    if not parts:
        parts.append("Generate a visualization from the available dataframes.")

    return "\n\n".join(parts)