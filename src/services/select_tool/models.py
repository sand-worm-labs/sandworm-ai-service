from typing import Any
from pydantic import BaseModel
from src.models.base import ChatContext, Message


class SelectToolRequest(BaseModel):
    messages: list[Message]
    model: str
    openrouter_api_key: str
    context: ChatContext


class SelectToolResponse(BaseModel):
    tool_id: str
    tool_name: str
    category: str
    viz: str | None
    inputs: list[Any]
    returns: list[Any]
    score: float
