from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Conversation(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[Message] = []


class CreateConversationRequest(BaseModel):
    title: str = "New Conversation"
    workspace_id: str
    user_id: str
    document_id: str
    openrouter_api_key: str


class UpdateConversationRequest(BaseModel):
    title: str


class CompletionRequest(BaseModel):
    messages: list[Message]
    model: str
    workspace_id: str
    user_id: str
    document_id: str
    openrouter_api_key: str
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int | None = None


class SendMessageRequest(BaseModel):
    content: str
    model: str
    workspace_id: str
    user_id: str
    document_id: str
    openrouter_api_key: str
    stream: bool = False
    max_tokens: int | None = None


class CompletionResponse(BaseModel):
    message: Message
    model: str
    usage: dict | None = None
