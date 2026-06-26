from __future__ import annotations

import uuid
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
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: BlockType
    title: str
    description: str
    content: str
    depends_on: list[int] = Field(default_factory=list)
