from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from src.models.base import BaseAiRequest, DocumentContext, Message


class Address(BaseModel):
    address: str
    chain: str


class Entity(BaseModel):
    addresses: list[Address] = Field(default_factory=list)
    protocol_names: list[str] = Field(default_factory=list)


class SubGoal(BaseModel):
    goal: str
    feasible: bool
    reason: str | None = None


class Intent(BaseModel):
    goal: str
    entity: Entity
    params: dict[str, Any] = Field(default_factory=dict)
    sub_goals: list[SubGoal] = Field(default_factory=list)


class Clarification(BaseModel):
    question: str
    recommendation: str
    missing_param: str


class ParseIntentRequest(BaseAiRequest):
    model: str
    context: DocumentContext
    history: list[Message] = Field(default_factory=list)