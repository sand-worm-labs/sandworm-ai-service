# ══════════════════════════════════════════════════════════════════════
# ⬢  BLOCK — TITLE
#    Handles all title generation scenarios:
#
#    FROM_CONTENT   no title yet → infer from document blocks
#    IMPROVE        existing title → suggest 3 richer variants
#    ITERATE        existing title + prior suggestions → suggest 3 more,
#                   none of which repeat anything already seen
#    CHAT_DIRECTED  user described tone/style in the chat panel →
#                   honour their request, still produce 3 options
#
#    Output contract:
#      FROM_CONTENT  → plain string (single title, no JSON)
#      all others    → JSON array of TitleSuggestion objects
#
#    TitleSuggestion shape:
#      { "title": str, "style": str, "why": str }
#
#    "style" is a short human-readable label the frontend can surface
#    e.g. "Analytical", "Punchy", "Question" so the user understands
#    what they're picking without reading the full title.
# ══════════════════════════════════════════════════════════════════════

from ..types import PromptContext, TitleMode, TitleTone


_TONE_GUIDE: dict[TitleTone, str] = {
    TitleTone.ANALYTICAL: (
        "Precise and metric-forward. Names the protocol/chain, the metric being "
        "measured, and optionally the time window or scope. "
        "Examples: 'Uniswap v3 LP Fee Revenue — 90d', "
        "'Base Wallet Concentration by Cohort', "
        "'Aave v3 Liquidation Events — Last 30 Days'"
    ),
    TitleTone.PUNCHY: (
        "Short, direct, optionally with a single relevant emoji. "
        "Reads like a dashboard card title, not a research paper. "
        "Examples: 'DEX Metrics 📊', 'Pump.Fun Alpha Wallets', "
        "'Bitcoin ETF 💼', 'Aerodrome Protocol Metrics ✈️'"
    ),
    TitleTone.QUESTION: (
        "Frames the analysis as a question the notebook answers. "
        "Creates curiosity and signals the analytical angle. "
        "Examples: 'Where Are Stablecoins Going?', "
        "'Who Makes the Most Volume on Base?', "
        "'Is This Pool Being Sandwiched?'"
    ),
    TitleTone.NARRATIVE: (
        "Tells a story or implies a finding. Works best when the "
        "notebook has a clear thesis or reveals something surprising. "
        "Examples: 'How MEV Bots Extracted $4M From This Pool', "
        "'The Wallets That Called the Depeg', "
        "'Tracking the Smart Money Before the Pump'"
    ),
    TitleTone.CASUAL: (
        "Relaxed, community-native tone. Lower case is fine. "
        "Reads like something a crypto-native analyst would post. "
        "Examples: 'perps & hyperliquid vibes', "
        "'memecoin wars', 'solana alpha wallets for copy-trading'"
    ),
}

_DEFAULT_TONE = TitleTone.ANALYTICAL

_FALLBACK_TITLES = [
    "Onchain Analysis",
]

_SYSTEM_FROM_CONTENT = """\
You generate precise, specific titles for onchain analytics notebooks.

A title should tell a reader immediately:
  - What is being measured (the metric or phenomenon)
  - What protocol, chain, or token it covers
  - Optionally: the time scope or analytical angle

### Style rules
- Default tone is analytical: specific, metric-forward, professional
- 3–10 words is the ideal range; shorter is almost always better
- Do NOT use generic phrases: "Analysis", "Deep Dive", "Overview" \
alone as the entire title — only acceptable if paired with something specific
- Emojis are acceptable if they feel natural, not forced
- Questions are acceptable if the notebook frames an investigation
- Do not start with "A", "An", or "The"
- Do not end with punctuation

### Output format
Return a **single plain string** — the title only. No JSON, no quotes, \
no explanation, no preamble.\
"""



def _build_system_suggest(mode: TitleMode) -> str:
    avoid_clause = (
        "\n\nIMPORTANT: The user has already seen these suggestions — "
        "do not produce any of them or close paraphrases:\n"
        if mode == TitleMode.ITERATE
        else ""
    )

    return f"""\
You generate title suggestions for onchain analytics notebooks.

You will receive an existing title (and optionally prior suggestions to avoid) \
and must produce exactly **3 distinct suggestions** that improve on it.

### Each suggestion must differ along a meaningful axis:
  - Different **length** (short / medium / longer descriptive)
  - Different **framing** (statement vs question vs punchy label)
  - Different **specificity level** (broad topic vs precise metric)

This gives the user a real choice, not three paraphrases of the same thing.

### Tone reference
{chr(10).join(f'- **{t.value.capitalize()}**: {g}' for t, g in _TONE_GUIDE.items())}

### Output format — return ONLY valid JSON, no other text:
```json
[
  {{
    "title": "The Suggested Title",
    "style": "Analytical",
    "why": "One sentence on what makes this version better or different"
  }},
  {{
    "title": "Another Suggestion",
    "style": "Punchy",
    "why": "..."
  }},
  {{
    "title": "Third Suggestion",
    "style": "Question",
    "why": "..."
  }}
]
```

Valid style labels: {', '.join(f'"{t.value.capitalize()}"' for t in TitleTone)}
{avoid_clause}\
"""



_SYSTEM_CHAT_DIRECTED = f"""\
You generate title suggestions for onchain analytics notebooks based on \
explicit user preferences expressed in the chat.

The user has told you what they want — tone, style, length, angle. \
Honour that request precisely. If they want an emoji, include one. \
If they want a question, phrase it as one. If they want it short, make it short.

Produce exactly **3 suggestions** that all satisfy the user's stated preference \
but vary from each other — different wording, different emphasis, \
different specificity.

### Tone reference (for when the user references a style by name)
{chr(10).join(f'- **{t.value.capitalize()}**: {g}' for t, g in _TONE_GUIDE.items())}

### Output format — return ONLY valid JSON, no other text:
```json
[
  {{
    "title": "The Suggested Title",
    "style": "Analytical",
    "why": "One sentence on how this satisfies the user's request"
  }}
]
```

Valid style labels: {', '.join(f'"{t.value.capitalize()}"' for t in TitleTone)}
"""



def _build_user_from_content(ctx: PromptContext) -> str:
    content = ctx.document_markdown.strip() or ctx.source.strip()

    if not content:
        import random
        return f"No document content available. Return this title: {random.choice(_FALLBACK_TITLES)}"

    preview   = content[:3000]
    truncated = len(content) > 3000

    tone = ctx.title_tone or _DEFAULT_TONE
    tone_instruction = (
        f"\nUse **{tone.value}** tone: {_TONE_GUIDE[tone]}"
        if ctx.title_tone
        else ""
    )

    lines = [f"**Document content:**\n\n{preview}"]
    if truncated:
        lines.append("_(content truncated — focus on what's above)_")
    if tone_instruction:
        lines.append(tone_instruction)

    return "\n\n".join(lines)


def _build_user_improve(ctx: PromptContext) -> str:
    lines: list[str] = []

    if ctx.existing_title:
        lines.append(f"**Current title:** {ctx.existing_title}")

    content = ctx.document_markdown.strip() or ctx.source.strip()
    if content:
        lines.append(f"**Document content (preview):**\n\n{content[:2000]}")

    if ctx.title_tone:
        lines.append(
            f"**Preferred tone:** {ctx.title_tone.value} — "
            f"{_TONE_GUIDE[ctx.title_tone]}"
        )

    lines.append("Suggest 3 improved titles. Return JSON only.")
    return "\n\n".join(lines)


def _build_user_iterate(ctx: PromptContext) -> str:
    lines: list[str] = []

    if ctx.existing_title:
        lines.append(f"**Current title:** {ctx.existing_title}")

    if ctx.previous_suggestions:
        already_seen = "\n".join(f"  - {s}" for s in ctx.previous_suggestions)
        lines.append(f"**Already suggested (do not repeat):**\n{already_seen}")

    content = ctx.document_markdown.strip() or ctx.source.strip()
    if content:
        lines.append(f"**Document content (preview):**\n\n{content[:2000]}")

    lines.append(
        "Suggest 3 **new** titles that differ from everything above. Return JSON only."
    )
    return "\n\n".join(lines)


def _build_user_chat_directed(ctx: PromptContext) -> str:
    lines: list[str] = []

    if ctx.title_user_context:
        lines.append(f"**User's request:** {ctx.title_user_context.strip()}")

    if ctx.existing_title:
        lines.append(f"**Current title:** {ctx.existing_title}")

    if ctx.previous_suggestions:
        already_seen = "\n".join(f"  - {s}" for s in ctx.previous_suggestions)
        lines.append(f"**Already suggested (avoid repeating):**\n{already_seen}")

    content = ctx.document_markdown.strip() or ctx.source.strip()
    if content:
        lines.append(f"**Document content (preview):**\n\n{content[:2000]}")

    lines.append("Suggest 3 titles that satisfy the user's request. Return JSON only.")
    return "\n\n".join(lines)



def build_directive(ctx: PromptContext) -> str:
    """System prompt — routed by title_mode."""
    match ctx.title_mode:
        case TitleMode.FROM_CONTENT:
            return _SYSTEM_FROM_CONTENT
        case TitleMode.IMPROVE:
            return _build_system_suggest(TitleMode.IMPROVE)
        case TitleMode.ITERATE:
            return _build_system_suggest(TitleMode.ITERATE)
        case TitleMode.CHAT_DIRECTED:
            return _SYSTEM_CHAT_DIRECTED
        case _:
            return _SYSTEM_FROM_CONTENT


def build_user_generate(ctx: PromptContext) -> str:
    """User message — routed by title_mode."""
    match ctx.title_mode:
        case TitleMode.FROM_CONTENT:
            return _build_user_from_content(ctx)
        case TitleMode.IMPROVE:
            return _build_user_improve(ctx)
        case TitleMode.ITERATE:
            return _build_user_iterate(ctx)
        case TitleMode.CHAT_DIRECTED:
            return _build_user_chat_directed(ctx)
        case _:
            return _build_user_from_content(ctx)