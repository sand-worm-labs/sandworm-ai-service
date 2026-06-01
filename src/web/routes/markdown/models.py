from pydantic import BaseModel
from src.models.base import BaseEditRequest, DocumentContext


class EditMarkdownRequest(BaseEditRequest):
    pass


class MarkdownResponse(BaseModel):
    content: str
    context: DocumentContext
