from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel


class ToolCallPart(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    toolName: str
    category: str
    params: dict[str, Any]



class ToolResultPart(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    summary: str
    rowCount: int | None = None
