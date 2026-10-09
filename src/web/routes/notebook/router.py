from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.models.base import DocumentContext
from src.services.agent.cell_edit import CellEditFailed
from src.services.notebook_title.service import NotebookTitleService

router = APIRouter()


class RenameRequest(BaseModel):
    openrouter_api_key: str
    model: str
    context: DocumentContext


class RenameResponse(BaseModel):
    """The rename itself was made by the MCP server; this says what the new title is."""

    title: str


@router.post("/title", response_model=RenameResponse)
async def rename_notebook(req: RenameRequest):
    try:
        title = await NotebookTitleService(req.openrouter_api_key, req.model, req.context).rename()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CellEditFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return RenameResponse(title=title)
