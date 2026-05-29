from pydantic import BaseModel
from src.models.base import BaseAiRequest, DocumentContext


class DocumentTitleRequest(BaseAiRequest):
    model: str
    context: DocumentContext


class DocumentTitleResponse(BaseModel):
    title: str
    context: DocumentContext
