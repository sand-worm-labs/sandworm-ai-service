from __future__ import annotations

from pydantic import BaseModel
from src.models.base import Message

class CompletionInput(BaseModel):
    messages: list[Message]
    model: str
    openrouter_api_key: str