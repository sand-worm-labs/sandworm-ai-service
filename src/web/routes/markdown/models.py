from pydantic import BaseModel
from src.models.base import DocumentContext


class EditMarkdownRequest(BaseModel):
    prompt: str
    openrouter_api_key: str
    model: str
    context: DocumentContext


class MarkdownResponse(BaseModel):
    content: str
    context: DocumentContext
