from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

from src.models.base import BaseAiRequest, ChatContext, DocumentContext, Message
from src.services.intent.models import Intent


BlockType = Literal["sql", "python", "visualization", "markdown"]


class PlannedBlock(BaseModel):
    type: BlockType
    title: str
    description: str
    depends_on: list[int] = Field(default_factory=list)


class BlockPlan(BaseModel):
    blocks: list[PlannedBlock]


class PlanBlocksRequest(BaseAiRequest):
    model: str
    intent: Intent
    context: DocumentContext | ChatContext
    history: list[Message] = Field(default_factory=list)
