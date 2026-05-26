# ══════════════════════════════════════════════════════════════════════
# ⬢  BLOCK — MARKDOWN
#    Two modes, many scenarios:
#
#    EDIT     — user is modifying an existing block (button or chat).
#               Has intents: fix / shorten / expand / rewrite /
#               reorganize / tone / custom.
#               May reference blocks above/below it.
#
#    GENERATE — block is being created during notebook generation.
#               Role-aware: intro / step / summary / insight / 
#               caption / section_header / conclusion / standalone.
#               Tone-aware: inherits notebook-level NotebookTone.
#               Context-aware: knows what block type precedes/follows.
# ══════════════════════════════════════════════════════════════════════

from ..types import (
    MarkdownBlockRole,
    MarkdownIntent,
    MarkdownMode,
    NotebookTone,
    PromptContext,
)




_MARKIT_SYNTAX = """\
Use only Markit-supported markdown syntax:
- Headings: `#` through `####` (H4 max — deeper nesting rarely helps in notebooks)
- Emphasis: `**bold**`, `_italic_`, `~~strikethrough~~`
- Lists: unordered (`-`) and ordered (`1.`), nestable one level
- Tables: GFM pipe syntax with alignment (`| :--- |`, `| ---: |`, `| :---: |`)
- Code: inline `` `code` `` and fenced blocks with language hint (` ```sql `, ` ```python `)
- Blockquotes: `>` for callouts, key findings, warnings
- Links: `[text](url)` — use for protocol docs, transaction links, Etherscan URLs
- Horizontal rule: `---` for section breaks within a block
- Task lists: `- [ ]` / `- [x]` for checklists and follow-up items\
"""


_INTENT_DIRECTIVE: dict[MarkdownIntent, str] = {
    MarkdownIntent.FIX: (
        "Fix grammar, spelling, and phrasing only. "
        "Preserve structure, heading hierarchy, all code blocks, tables, "
        "and every numerical value exactly. Do not change any numbers, units, "
        "or technical terms."
    ),
    MarkdownIntent.SHORTEN: (
        "Reduce length by cutting filler, repetition, and over-explanation. "
        "Every number, table, and code block must survive intact. "
        "Cut prose, not substance. If a sentence doesn't add analytical value, remove it."
    ),
    MarkdownIntent.EXPAND: (
        "Add depth and analytical detail. Elaborate on the blockchain/DeFi mechanics "
        "involved, add relevant context, surface implications of the data. "
        "Do not invent statistics or claim findings that aren't supported by the content. "
        "If context blocks are provided, you may reference their findings."
    ),
    MarkdownIntent.REWRITE: (
        "Rewrite completely for clarity and precision. Preserve all data points, "
        "findings, and code blocks. Aim for the directness of a quant research note: "
        "findings first, explanation second, no hedging, no padding."
    ),
    MarkdownIntent.REORGANIZE: (
        "Restructure the content for better logical flow. Reorder sections if needed, "
        "promote or demote heading levels, break long paragraphs, consolidate related "
        "points. Do not add or remove information — only reorganise what's there."
    ),
    MarkdownIntent.TONE: (
        ""  # filled dynamically — requires ctx.instructions to name the target tone
    ),
    MarkdownIntent.CUSTOM: (
        ""  # filled dynamically from ctx.instructions
    ),
}



_TONE_GUIDE: dict[NotebookTone, str] = {
    NotebookTone.TECHNICAL_BLOG: (
        "Assumes a crypto-native reader. Use protocol names, mechanism names, "
        "and DeFi terminology freely without inline explanation. Dense but readable. "
        "Think Paradigm research post or Delphi Digital report."
    ),
    NotebookTone.ACCESSIBLE_BLOG: (
        "Written for a curious reader who understands crypto broadly but may not know "
        "the specific mechanism. Explain jargon inline, use analogies where helpful. "
        "Never condescending — treat the reader as intelligent, just less specialised."
    ),
    NotebookTone.RESEARCH_NOTE: (
        "Quant analyst style. Findings-first, dense prose, minimal scene-setting. "
        "Numbers and percentages are primary; narrative is secondary. "
        "Short sentences. No unnecessary context. Think Bloomberg terminal note."
    ),
    NotebookTone.STEP_BY_STEP: (
        "Tutorial/walkthrough style. Explains each analytical decision as it's made. "
        "Addresses the reader directly ('we first filter by...', 'notice that...'). "
        "Comfortable with longer explanations if they build understanding."
    ),
    NotebookTone.CASUAL: (
        "Community post style. Direct, opinionated, may use crypto-native slang. "
        "Short paragraphs. Reads like a good Twitter/Farcaster thread written out. "
        "No corporate voice. Opinions are welcome."
    ),
}



_ROLE_GUIDE: dict[MarkdownBlockRole, tuple[str, str]] = {
    # (purpose, length_hint)
    MarkdownBlockRole.INTRO: (
        "Opens the notebook. Establishes context, states the analytical question, "
        "explains why it matters. May reference the protocol, chain, or time period "
        "being studied. Sets expectations for what the notebook will show.",
        "2–4 paragraphs. Long enough to orient the reader, short enough to not delay "
        "getting to the data.",
    ),
    MarkdownBlockRole.STEP: (
        "Describes what the immediately following code or query block does. "
        "Explains the analytical approach and any non-obvious decisions. "
        "Should prepare the reader to understand the output before they see it.",
        "1–3 sentences to 1 short paragraph. Be concise — the code speaks for itself.",
    ),
    MarkdownBlockRole.SUMMARY: (
        "Summarises what the preceding block's results show. Calls out the key number, "
        "trend, or pattern. Does not just restate the query — interprets the output.",
        "1–2 paragraphs. Lead with the headline finding.",
    ),
    MarkdownBlockRole.INSIGHT: (
        "Goes beyond the data to add analytical colour. Raises implications, "
        "connects findings to protocol mechanics or market context, flags anomalies. "
        "May suggest follow-up questions. Can reference other blocks in the notebook.",
        "1–3 paragraphs. This is where analytical judgment lives.",
    ),
    MarkdownBlockRole.CAPTION: (
        "Short description directly below a visualization. Names the chart, "
        "highlights the most important thing to notice in it, and optionally "
        "notes a limitation or caveat in the data.",
        "1–3 sentences. No more.",
    ),
    MarkdownBlockRole.SECTION_HEADER: (
        "Transitions between major sections of the notebook. Briefly closes out "
        "the section just finished and previews what the next section covers.",
        "2–4 sentences. Light touch — don't over-explain the transition.",
    ),
    MarkdownBlockRole.CONCLUSION: (
        "Closes the notebook. Restates the key findings, draws conclusions, notes "
        "limitations and caveats, suggests next steps or follow-on analyses. "
        "Does not introduce new data or claims.",
        "2–5 paragraphs depending on notebook complexity.",
    ),
    MarkdownBlockRole.STANDALONE: (
        "Independent markdown block. Infer the appropriate purpose and length from "
        "the notebook intent, surrounding blocks, and content being written.",
        "Match the depth of surrounding content.",
    ),
}


_BLOCK_TYPE_LABEL: dict[str, str] = {
    "sql":      "SQL query",
    "python":   "Python code block",
    "viz":      "visualization",
    "markdown": "markdown block",
}



def _build_system_edit(ctx: PromptContext) -> str:
    intent = ctx.markdown_intent

    if intent == MarkdownIntent.TONE:
        directive = (
            f"Change the tone to: **{ctx.instructions.strip()}**.\n"
            "Rewrite the content preserving all information and structure, "
            "adjusting only the voice, register, and phrasing to match the requested tone."
        )
    elif intent == MarkdownIntent.CUSTOM:
        directive = ctx.instructions.strip() or "(no instruction provided)"
    else:
        directive = _INTENT_DIRECTIVE[intent]

    has_context = bool(ctx.preceding_block_content or ctx.neighbor_blocks)
    context_note = (
        "\nContext from other blocks is provided in the user message. "
        "You may reference specific values or findings from them if the "
        "instruction implies it."
        if has_context else ""
    )

    return (
        f"## Task: Edit Markdown Block\n\n"
        f"You are editing a text block in an onchain analytics notebook. "
        f"Content may include analysis prose, data tables, code, and quantitative findings.\n\n"
        f"{_MARKIT_SYNTAX}\n\n"
        f"Return **only** the edited markdown — no explanation, no preamble, "
        f"no surrounding fences.{context_note}\n\n"
        f"**Edit directive:** {directive}"
    )



def _build_system_generate(ctx: PromptContext) -> str:
    role         = ctx.markdown_block_role
    tone         = ctx.notebook_tone
    purpose, length_hint = _ROLE_GUIDE[role]

    tone_section = ""
    if tone:
        tone_section = (
            f"\n\n**Notebook tone — {tone.value.replace('_', ' ')}:** "
            f"{_TONE_GUIDE[tone]}"
        )

    preceding_note = ""
    if ctx.preceding_block_type:
        label = _BLOCK_TYPE_LABEL.get(ctx.preceding_block_type, ctx.preceding_block_type)
        preceding_note = (
            f"\n\nThe block immediately **above** this is a {label}. "
            f"{'Its content/results are provided in the user message — reference specific findings.' if ctx.preceding_block_content else ''}"
        )

    following_note = ""
    if ctx.following_block_type:
        label = _BLOCK_TYPE_LABEL.get(ctx.following_block_type, ctx.following_block_type)
        following_note = (
            f"\nThe block immediately **below** this will be a {label}. "
            f"{'Transition toward it.' if role == MarkdownBlockRole.STEP else 'Keep that in mind for flow.'}"
        )

    return (
        f"## Task: Generate Markdown Block\n\n"
        f"You are writing a **{role.value.replace('_', ' ')}** block "
        f"in an onchain analytics notebook.\n\n"
        f"**Block purpose:** {purpose}\n\n"
        f"**Length guidance:** {length_hint}"
        f"{tone_section}"
        f"{preceding_note}"
        f"{following_note}\n\n"
        f"{_MARKIT_SYNTAX}\n\n"
        f"Return **only** the markdown — no explanation, no preamble, no surrounding fences."
    )



def _build_user_edit(ctx: PromptContext) -> str:
    parts: list[str] = []

    if ctx.source.strip():
        parts.append(f"**Content to edit:**\n\n{ctx.source.strip()}")

    if ctx.markdown_intent == MarkdownIntent.CUSTOM and ctx.instructions:
        parts.append(f"**Instruction:** {ctx.instructions.strip()}")

    if ctx.preceding_block_content:
        label = _BLOCK_TYPE_LABEL.get(ctx.preceding_block_type, "preceding block")
        parts.append(
            f"**Context — {label} above:**\n\n"
            f"```\n{ctx.preceding_block_content.strip()[:1500]}\n```"
        )

    if ctx.neighbor_blocks:
        neighbors = "\n\n".join(ctx.neighbor_blocks[:3])
        parts.append(f"**Other notebook context:**\n\n{neighbors}")

    return "\n\n".join(parts)


def _build_user_generate(ctx: PromptContext) -> str:
    parts: list[str] = []

    if ctx.notebook_intent:
        parts.append(f"**Notebook intent:** {ctx.notebook_intent.strip()}")

    if ctx.instructions:
        parts.append(f"**Specific instruction for this block:** {ctx.instructions.strip()}")

    if ctx.preceding_block_content:
        label = _BLOCK_TYPE_LABEL.get(ctx.preceding_block_type, "preceding block")
        parts.append(
            f"**{label.capitalize()} above (content/results):**\n\n"
            f"```\n{ctx.preceding_block_content.strip()[:2000]}\n```"
        )

    if ctx.neighbor_blocks:
        neighbors = "\n\n".join(ctx.neighbor_blocks[:4])
        parts.append(f"**Other blocks in notebook:**\n\n{neighbors}")

    if not parts:
        parts.append("Generate the markdown block based on the role and context above.")

    return "\n\n".join(parts)



def build_directive(ctx: PromptContext) -> str:
    if ctx.markdown_mode == MarkdownMode.GENERATE:
        return _build_system_generate(ctx)
    return _build_system_edit(ctx)


def build_user_edit(ctx: PromptContext) -> str:
    if ctx.markdown_mode == MarkdownMode.GENERATE:
        return _build_user_generate(ctx)
    return _build_user_edit(ctx)