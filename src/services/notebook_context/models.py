from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AiContextRequest:
    document_id: str
    workspace_id: str
    focused_block_ids: list[str] | None = field(default=None)