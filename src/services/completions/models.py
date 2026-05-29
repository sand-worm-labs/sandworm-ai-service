from __future__ import annotations

from pydantic import BaseModel
from src.models.base import ChatContext, Message

class CompletionRequest(BaseModel):
    messages: list[Message]
    model: str
    openrouter_api_key: str
    context: ChatContext
    derived_context: str | None = None
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int | None = None