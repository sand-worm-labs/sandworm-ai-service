from fastapi import APIRouter, HTTPException

from src.models.base import BaseEditRequest, BaseFixRequest, CellEditResponse
from src.services.agent.cell_edit import CellEditFailed
from src.services.cell.service import CellKind, CellService


def make_cell_router(kind: CellKind) -> APIRouter:
    """Edit (and, for code/sql, fix) endpoints for one cell type. The cell is changed by the MCP server."""
    router = APIRouter()

    async def run(action: str, req, text: str) -> CellEditResponse:
        service = CellService(req.openrouter_api_key, req.model, kind, req.context, req.block_id)
        try:
            await getattr(service, action)(text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except CellEditFailed as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return CellEditResponse(cell_id=req.block_id, updated=True)

    @router.post("/edit", response_model=CellEditResponse)
    async def edit_cell(req: BaseEditRequest):
        return await run("edit", req, req.prompt)

    if kind != "markdown":

        @router.post("/fix", response_model=CellEditResponse)
        async def fix_cell(req: BaseFixRequest):
            return await run("fix", req, req.error_message)

    return router


code_router = make_cell_router("python")
sql_router = make_cell_router("sql")
markdown_router = make_cell_router("markdown")
