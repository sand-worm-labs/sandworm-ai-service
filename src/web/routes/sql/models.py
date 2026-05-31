from pydantic import BaseModel
from src.models.base import DocumentContext


class EditSqlRequest(BaseModel):
    prompt: str
    openrouter_api_key: str
    model: str
    context: DocumentContext


class FixSqlRequest(BaseModel):
    error_message: str
    openrouter_api_key: str
    model: str
    context: DocumentContext


class SqlResponse(BaseModel):
    code: str
    context: DocumentContext
