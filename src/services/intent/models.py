from typing import Any
from pydantic import BaseModel
from src.models.base import DocumentContext


class Intent(BaseModel):
    """Structured representation of a user's intent inside a document."""

    goal: str
    """The action the user wants to perform.

    Common values: ``summarize``, ``rewrite``, ``translate``, ``expand``,
    ``shorten``, ``analyze``, ``explain``, ``extract``, ``format``,
    ``generate``, ``compare``, ``search``, ``edit``, ``fix``.
    """

    entity: str
    """The document fragment being acted on.

    Common values: ``document``, ``paragraph``, ``selection``, ``heading``,
    ``table``, ``code``, ``list``, ``sentence``, ``title``.
    """

    params: dict[str, Any] = {}
    """Free-form key/value pairs extracted from the message.

    Examples: ``{"language": "French"}``, ``{"tone": "formal"}``,
    ``{"length": "short"}``, ``{"format": "bullet_points"}``.
    """


class ParseIntentRequest(BaseModel):
    """Request body for ``POST /document/parse-intent``."""

    message: str
    openrouter_api_key: str
    context: DocumentContext
    model: str = "anthropic/claude-haiku-4-5-20251001"


class ParseIntentResponse(BaseModel):
    """Response body returned by ``POST /document/parse-intent``."""

    intent: Intent
    context: DocumentContext
