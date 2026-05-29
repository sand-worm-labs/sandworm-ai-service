from __future__ import annotations

from pydantic import BaseModel
from src.models.base import ChatContext, Message

class CompletionRequest(BaseModel):
    """Request body for the /chat/completions endpoint."""

    messages: list[Message]
    model: str
    openrouter_api_key: str
    context: ChatContext
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int | None = None