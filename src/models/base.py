from typing import Literal
from pydantic import BaseModel


class BaseContext(BaseModel):
    """Shared context fields present on every AI request."""

    user_id: str
    workspace_id: str
    document_id: str


class ChatContext(BaseContext):
    """Extends base context with chat session id."""

    chat_id: str | None = None


class DocumentContext(BaseContext):
    """Document-scoped context — no chat_id."""


class BaseAiRequest(BaseModel):
    """Minimal fields required by all single-message AI requests."""

    message: str
    openrouter_api_key: str


class Message(BaseModel):
    """A single turn in a conversation thread."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class Usage(BaseModel):
    """Token-usage breakdown returned by the model provider."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
