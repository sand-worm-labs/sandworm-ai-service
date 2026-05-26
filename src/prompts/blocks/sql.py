from ..types import PromptContext, SQLDialect


def build_directive_edit(ctx: PromptContext) -> str:
    dialect_note = _dialect_note(ctx.dialect)
    return f"""\
## Task: Edit SQL Query

Apply the user's instructions to the existing query. Return **only** the \
modified SQL — no explanation, no markdown fences, no preamble.

Rules:
- Preserve the original query's intent unless the instruction explicitly changes it
- Preserve existing inline comments unless they are factually wrong after the edit
- Make the minimal diff needed to satisfy the instruction
- {dialect_note}
- If the instruction is ambiguous, resolve it in the most analytically useful way\
"""


def build_directive_fix(ctx: PromptContext) -> str:
    dialect_note = _dialect_note(ctx.dialect)
    return f"""\
## Task: Fix SQL Query

The query produced an error. Diagnose and fix it. Return **only** the \
corrected SQL — no explanation, no markdown fences, no preamble.

Rules:
- Fix only what is broken — do not refactor unrelated parts
- {dialect_note}
- If the error is a missing column, check the schema before inventing one
- If the fix requires a structural change (subquery, CTE, type cast), make it\
"""


def build_directive_generate(ctx: PromptContext) -> str:
    dialect_note = _dialect_note(ctx.dialect)
    return f"""\
## Task: Generate SQL Query

Write a SQL query from scratch that satisfies the user's analytical intent. \
Return **only** the SQL — no explanation, no markdown fences, no preamble.

Rules:
- {dialect_note}
- Use CTEs for multi-step logic — prefer readability over golf
- Include inline comments on non-obvious steps (one line max each)
- Only reference tables and columns that exist in the provided schema, \
or are standard Dune-style tables if no schema is given
- Normalise token amounts: `CAST(raw AS DOUBLE) / POW(10, decimals)` \
— never integer division on raw on-chain values
- If Grimoire templates were provided, use them as the foundation and adapt \
to the specific intent rather than starting from scratch\
"""


def build_user_edit(ctx: PromptContext) -> str:
    return "\n\n".join(filter(None, [
        f"**Dialect:** `{ctx.dialect}`",
        f"**Current query:**\n```sql\n{ctx.source.strip()}\n```",
        f"**Instructions:** {ctx.instructions.strip()}",
    ]))


def build_user_fix(ctx: PromptContext) -> str:
    return "\n\n".join(filter(None, [
        f"**Dialect:** `{ctx.dialect}`",
        f"**Current query:**\n```sql\n{ctx.source.strip()}\n```",
        f"**Error:**\n```\n{ctx.error_message.strip()}\n```",
        "Fix the error. Return only the corrected SQL.",
    ]))


def build_user_generate(ctx: PromptContext) -> str:
    return "\n\n".join(filter(None, [
        f"**Dialect:** `{ctx.dialect}`",
        f"**Intent:** {ctx.instructions.strip()}",
    ]))


def _dialect_note(dialect: SQLDialect) -> str:
    if dialect == SQLDialect.DUCKDB:
        return (
            "Use DuckDB dialect: `INTERVAL '7 days'`, `DATE_TRUNC`, "
            "`APPROX_COUNT_DISTINCT`, `list_agg`, `struct_pack` where beneficial"
        )
    return "Use ANSI-compatible SQL — avoid DuckDB-specific extensions"