from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


NotebookContextMode = Literal["none", "spine", "full"]


@dataclass
class AiContextRequest:
    document_id: str
    workspace_id: str
    focused_block_ids: list[str] | None = field(default=None)