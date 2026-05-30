from __future__ import annotations

from typing import Literal
from pydantic import BaseModel


class BlockActionPart(BaseModel):
    type: Literal["block_action"] = "block_action"
    action: Literal["created", "ran", "edited"]
    blockType: str
    blockTitle: str
    blockId: str
