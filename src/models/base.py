from typing import Literal

from pydantic import BaseModel


class BaseContext(BaseModel):
    """Shared context fields present on every AI request."""

    user_id: str
    workspace_id: str
    document_id: str
    focused_block_ids: list[str] | None = None
    # The user's Sandworm access token. Everything this service does to a
    # notebook goes through the MCP server with it, so tools act as this user.
    user_token: str | None = None
    # False when neither Dune nor Sandworm Cloud can run SQL; notebooks are then
    # built from public APIs.
    sql_available: bool = True
    # True when chain SQL is off because the workspace is on the free plan.
    paid_plan_required: bool = False


class ChatContext(BaseContext):
    """Extends base context with chat session id."""

    chat_id: str

class DocumentContext(BaseContext):
    """Document-scoped context. chat_id is set when the edit was made from a chat, so the AI service can stream the result to it."""

    chat_id: str | None = None


class BaseAiRequest(BaseModel):
    """Minimal fields required by all single-message AI requests."""

    message: str
    openrouter_api_key: str


class BaseEditRequest(BaseModel):
    prompt: str
    block_id: str
    openrouter_api_key: str
    model: str
    context: DocumentContext


class BaseFixRequest(BaseModel):
    error_message: str
    block_id: str
    openrouter_api_key: str
    model: str
    context: DocumentContext


class CellEditResponse(BaseModel):
    """The edit itself was made by the MCP server; this only says it happened."""

    cell_id: str
    updated: bool


class Message(BaseModel):
    """A single turn in a conversation thread."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class Usage(BaseModel):
    """Token-usage breakdown returned by the model provider."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
