from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.models.base import BaseAiRequest, ChatContext, DocumentContext, Message

# params keys the intent prompt's array convention applies to — a bare
# value here gets coerced to a single-element list rather than rejected,
# since Intent.model_validate() runs with no retry/repair path around it
# (see pipeline/service.py's node_plan_blocks / node_generate_blocks).
_MULTI_VALUE_PARAM_KEYS = frozenset({"chains"})


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

    @field_validator("params")
    @classmethod
    def _coerce_multi_value_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        for key in _MULTI_VALUE_PARAM_KEYS:
            value = params.get(key)
            if isinstance(value, str):
                params[key] = [value]
        return params


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