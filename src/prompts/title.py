from ..types import PromptContext


def build_directive(ctx: PromptContext) -> str:
    return """\
## Task: Generate Notebook Title

Read the document content and produce a single, precise title that captures \
the core analytical question or finding.

Output contract — strictly enforced:
- **One line only** — the title, nothing else
- **No quotes**, no punctuation at the end, no markdown, no preamble
- **5–9 words** is the ideal range; do not pad to hit a word count
- **Specific over generic**: "Uniswap v3 USDC/ETH LP Fee Revenue — 90d" \
beats "DeFi Protocol Analysis"
- Use onchain analytics vocabulary: wallets, protocols, tokens, chains, \
time windows, metrics — not generic business language\
"""


def build_user_generate(ctx: PromptContext) -> str:
    content = ctx.document_markdown.strip() or ctx.source.strip()

    if not content:
        return "Generate a title for an onchain analytics notebook."

    # Trim to avoid burning tokens on very long docs — the first ~2000
    # chars capture the block types, headings, and opening content which
    # is sufficient for title generation.
    truncated = content[:2000]
    suffix    = "…\n_(content truncated)_" if len(content) > 2000 else ""

    return f"**Document content:**\n\n{truncated}{suffix}"