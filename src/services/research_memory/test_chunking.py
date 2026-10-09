from __future__ import annotations

from src.services.research_memory import chunking


def test_chunks_lead_with_the_title_and_respect_the_size():
    text = "\n\n".join(f"paragraph {i} " + "x" * 400 for i in range(10))

    chunks = chunking.chunk_notebook("TVL study", text)

    assert len(chunks) > 1
    assert all(c.startswith("TVL study\n\n") for c in chunks)
    assert all(len(c) <= chunking.CHUNK_CHARS + len("TVL study\n\n") for c in chunks)


def test_one_huge_block_is_cut_and_the_chunk_count_is_capped():
    chunks = chunking.chunk_notebook("t", "y" * (chunking.CHUNK_CHARS * (chunking.MAX_CHUNKS + 20)))

    assert len(chunks) == chunking.MAX_CHUNKS
