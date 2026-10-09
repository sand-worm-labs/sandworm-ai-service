from __future__ import annotations

CHUNK_CHARS = 1500
MAX_CHUNKS = 200


def chunk_notebook(title: str, markdown: str) -> list[str]:
    """Pack paragraphs into chunks of about CHUNK_CHARS, each led by the title."""
    chunks: list[str] = []
    current = ""
    for block in (b.strip() for b in markdown.split("\n\n")):
        if not block:
            continue
        for start in range(0, len(block), CHUNK_CHARS):  # a single huge block is cut
            piece = block[start : start + CHUNK_CHARS]
            if current and len(current) + len(piece) + 2 > CHUNK_CHARS:
                chunks.append(current)
                current = piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return [f"{title}\n\n{c}" for c in chunks[:MAX_CHUNKS]]
