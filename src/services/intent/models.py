from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.models.base import BaseAiRequest, ChatContext, DocumentContext, Message


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


class ClarifyOption(BaseModel):
    label: str
    value: str


class Clarification(BaseModel):
    question: str
    recommendation: str
    missing_param: str
    options: list[ClarifyOption] = Field(default_factory=list)


class ParseIntentRequest(BaseAiRequest):
    model: str
    context: DocumentContext | ChatContext
    history: list[Message] = Field(default_factory=list)
    job_id: str | None = None


class IntentClass(str, Enum):
    ANALYTICAL     = "analytical"
    CONVERSATIONAL = "conversational"
    EXPLANATORY    = "explanatory"
    EDITORIAL      = "editorial"


@dataclass
class ParsedIntent:
    intent_class: IntentClass
    intent_status: Literal["clarify", "complete", "error"]
    intent: dict[str, Any] | None = None
    references_block: bool = False

    @property
    def is_complete(self) -> bool:
        return self.intent_status == "complete"