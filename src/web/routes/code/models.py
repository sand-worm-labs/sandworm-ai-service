from pydantic import BaseModel
from src.models.base import DocumentContext


class EditCodeRequest(BaseModel):
    prompt: str
    openrouter_api_key: str
    model: str
    context: DocumentContext


class FixCodeRequest(BaseModel):
    error_message: str
    openrouter_api_key: str
    model: str
    context: DocumentContext


class CodeResponse(BaseModel):
    code: str
    context: DocumentContext
