from ..types import MarkdownIntent, PromptContext


_INTENT_DIRECTIVE: dict[MarkdownIntent, str] = {
    MarkdownIntent.FIX: (
        "Fix grammar, spelling, and phrasing. "
        "Preserve the structure, heading hierarchy, all code blocks, "
        "and all numerical values exactly — do not change any numbers or units."
    ),
    MarkdownIntent.SHORTEN: (
        "Reduce the length. Remove filler, repetition, and over-explanation. "
        "Every number, table, and code block must survive. "
        "Cut prose, not substance."
    ),
    MarkdownIntent.EXPAND: (
        "Add depth and detail. Elaborate on analytical implications, add relevant "
        "context about the blockchain / DeFi mechanics involved, suggest follow-up "
        "angles. Do not invent data or statistics that weren't in the original. "
        "Preserve all existing structure."
    ),
    MarkdownIntent.REWRITE: (
        "Rewrite completely for clarity and precision. "
        "Preserve the original meaning, all data points, all code blocks. "
        "Aim for the directness of a quant research note: no hedging, no padding."
    ),
    MarkdownIntent.CUSTOM: "",  
}

def build_directive(ctx: PromptContext) -> str:
    intent    = ctx.markdown_intent
    directive = _INTENT_DIRECTIVE[intent]

    if intent == MarkdownIntent.CUSTOM:
        directive = ctx.instructions.strip() if ctx.instructions else "(no instruction provided)"

    return f"""\
## Task: Edit Markdown Block

You are editing a text block in an onchain analytics notebook. \
The content may include analysis prose, data tables, code snippets, \
and quantitative findings.

Return **only** the edited markdown — no explanation, no preamble, \
no surrounding code fences.

**Edit directive:** {directive}\
"""



def build_user_edit(ctx: PromptContext) -> str:
    parts = [f"**Content to edit:**\n\n{ctx.source.strip()}"]

    if ctx.markdown_intent == MarkdownIntent.CUSTOM and ctx.instructions:
        parts.append(f"**Instruction:** {ctx.instructions.strip()}")

    return "\n\n".join(parts)