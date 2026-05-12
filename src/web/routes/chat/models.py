from typing import Literal
from pydantic import BaseModel

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class RequestContext(BaseModel):
    user_id: str
    workspace_id: str
    document_id: str
    chat_id: str= None

class CompletionRequest(BaseModel):
    messages: list[Message]
    model: str
    openrouter_api_key: str
    context: RequestContext
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int | None = None

class CompletionResponse(BaseModel):
    message: Message
    model: str
    context: RequestContext
    finish_reason: str | None = None
    generation_id: str | None = None
    usage: Usage | None = None

class TitleRequest(BaseModel):
    message: str
    openrouter_api_key: str
    context: RequestContext

class TitleResponse(BaseModel):
    title: str
    context: RequestContext