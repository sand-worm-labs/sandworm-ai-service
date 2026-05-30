from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class BlockActionPart(BaseModel):
    type: Literal["block_action"] = "block_action"
    action: Literal["created", "ran", "edited"]
    blockType: str
    blockTitle: str
    blockId: str


BlockType = Literal["sql", "python", "visualization", "markdown"]


class GeneratedBlock(BaseModel):
    type: BlockType
    title: str
    description: str
    content: str
    depends_on: list[int] = Field(default_factory=list)
