from __future__ import annotations

from dataclasses import dataclass

NOTEBOOK = "notebook"
ATTEMPT = "attempt"


@dataclass
class Attempt:
    """One thing the AI tried, and how it went."""

    tool: str
    summary: str
    # None when nobody has run it yet: a suggested edit or fix that the user
    # has not accepted, so it is neither a success nor a failure.
    worked: bool | None
    result: str

    def text(self) -> str:
        outcome = "SUGGESTED" if self.worked is None else "WORKED" if self.worked else "FAILED"
        return f"{outcome}: {self.tool} {self.summary}\nResult: {self.result}".strip()


@dataclass
class MemoryHit:
    kind: str
    document_id: str
    title: str
    text: str
    score: float
    worked: bool | None = None
