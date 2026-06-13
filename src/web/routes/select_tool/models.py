from typing import Any
from pydantic import BaseModel
from src.models.base import ChatContext


class SelectToolRequest(BaseModel):
    message: str
    openrouter_api_key: str
    model: str
    context: ChatContext


class SelectToolResponse(BaseModel):
    tool_name: str
    category: str
    params: dict[str, Any]
