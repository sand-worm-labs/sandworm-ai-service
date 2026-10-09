from __future__ import annotations

import re
import zlib
from collections import Counter

from qdrant_client.models import SparseVector

# Notebook memory is searched by words, not by a model: every word is hashed to
# a number, and a text becomes a sparse vector of word weights. Qdrant scores a
# search with its IDF modifier on the collection, so rare words count for more
# than common ones: this is BM25. There is nothing to download or load, and a
# page of text is turned into a vector in well under a millisecond.

# Keeps addresses, hashes, tickers and snake_case names whole (0xabc..., usdc, tvl_usd).
_WORD = re.compile(r"[a-z0-9]+(?:[_.'-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a about all also an and any are as at be been but by can could did do does for from had has have how i if in "
    "into is it its just me more most my no not of on one or our out over so some such than that the their them "
    "then there these they this to up us was we were what when where which who why will with would you your".split()
)
MAX_WORD_CHARS = 64

# BM25's usual constants: how fast repeats of a word stop counting, and how
# much a long text is held against a short one. AVG_WORDS is a typical chunk.
K1 = 1.2
B = 0.75
AVG_WORDS = 120


def words(text: str) -> list[str]:
    found = (w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) <= MAX_WORD_CHARS)
    # Also index the parts of a compound (tvl_usd -> tvl, usd) so a search for either finds it.
    out: list[str] = []
    for word in found:
        out.append(word)
        parts = re.split(r"[_.'-]", word)
        if len(parts) > 1:
            out.extend(p for p in parts if p and p not in _STOPWORDS)
    return out


def _index(word: str) -> int:
    return zlib.crc32(word.encode())


def _vector(weights: dict[int, float]) -> SparseVector:
    indices = sorted(weights)
    return SparseVector(indices=indices, values=[weights[i] for i in indices])


def document_vector(text: str) -> SparseVector:
    """A stored text: each word weighted by how often it appears, the way BM25 does."""
    counts = Counter(words(text))
    length = sum(counts.values())
    norm = K1 * (1 - B + B * length / AVG_WORDS)
    weights: dict[int, float] = {}
    for word, count in counts.items():
        weight = count * (K1 + 1) / (count + norm)
        i = _index(word)
        weights[i] = weights.get(i, 0.0) + weight  # two words sharing a hash just add up
    return _vector(weights)


def query_vector(text: str) -> SparseVector:
    """A search: each distinct word counts once; Qdrant multiplies in how rare it is."""
    return _vector({_index(word): 1.0 for word in set(words(text))})
