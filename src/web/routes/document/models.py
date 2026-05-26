from pydantic import BaseModel
from src.models.base import BaseAiRequest, DocumentContext


class DocumentTitleRequest(BaseAiRequest):
    """Request body for the /document/generate-title endpoint."""

    context: DocumentContext


class DocumentTitleResponse(BaseModel):
    """Response body returned by the /document/generate-title endpoint."""

    title: str
    context: DocumentContext
