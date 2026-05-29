from pydantic import BaseModel
from src.models.base import BaseAiRequest, ChatContext, Message, Usage

# Re-export shared types so existing imports keep working.
__all__ = [
    "Message",
    "Usage",
    "ChatContext",
    "CompletionRequest",
    "CompletionResponse",
    "TitleRequest",
    "TitleResponse",
]


class CompletionRequest(BaseModel):
    """Request body for the /chat/completions endpoint."""

    messages: list[Message]
    model: str
    openrouter_api_key: str
    context: ChatContext
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int | None = None


class CompletionResponse(BaseModel):
    """Response body returned by the /chat/completions endpoint."""

    message: Message
    model: str
    context: ChatContext
    finish_reason: str | None = None
    generation_id: str | None = None
    usage: Usage | None = None


class TitleRequest(BaseAiRequest):
    """Request body for the /chat/title endpoint."""

    context: ChatContext


class TitleResponse(BaseModel):
    """Response body returned by the /chat/title endpoint."""

    title: str
    context: ChatContext
