from __future__ import annotations

from typing import Literal
from pydantic import BaseModel


class NotebookBlockResultEvent(BaseModel):
    type: Literal["block_result"]
    documentId: str
    workspaceId: str
    blockId: str
    blockType: str
    status: Literal["success", "error"]
    error: str | None = None


NOTEBOOK_EVENTS_CHANNEL = "notebook:events"
