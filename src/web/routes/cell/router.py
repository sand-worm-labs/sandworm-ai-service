import asyncio

from fastapi import APIRouter, HTTPException, Request

from src.models.base import BaseEditRequest, BaseFixRequest, CellEditResponse
from src.services.agent.cell_edit import CellEditFailed
from src.services.cell.service import CellKind, CellService

DISCONNECT_POLL_SECONDS = 0.25
# nginx's "client closed request": nobody is left to read the answer.
CLIENT_CLOSED_REQUEST = 499


async def _disconnected(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(DISCONNECT_POLL_SECONDS)


async def _unless_stopped(request: Request, work) -> None:
    """Run the edit, and drop it the moment the API closes the request.

    That is how a user stops an edit: the API aborts its call here. The model
    call and the cell write are cancelled wherever they are, so a stopped edit
    that has not yet reached update_cell changes nothing.
    """
    task = asyncio.create_task(work)
    watcher = asyncio.create_task(_disconnected(request))
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        stopped = not task.done()
    finally:
        watcher.cancel()
        task.cancel()
    if stopped:
        await asyncio.gather(task, return_exceptions=True)
        raise HTTPException(status_code=CLIENT_CLOSED_REQUEST, detail="The edit was stopped.")
    task.result()


def make_cell_router(kind: CellKind) -> APIRouter:
    """Edit (and, for code/sql, fix) endpoints for one cell type. The cell is changed by the MCP server."""
    router = APIRouter()

    async def run(action: str, req, text: str, request: Request) -> CellEditResponse:
        service = CellService(req.openrouter_api_key, req.model, kind, req.context, req.block_id)
        try:
            await _unless_stopped(request, getattr(service, action)(text))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except CellEditFailed as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return CellEditResponse(cell_id=req.block_id, updated=True)

    @router.post("/edit", response_model=CellEditResponse)
    async def edit_cell(req: BaseEditRequest, request: Request):
        return await run("edit", req, req.prompt, request)

    if kind != "markdown":

        @router.post("/fix", response_model=CellEditResponse)
        async def fix_cell(req: BaseFixRequest, request: Request):
            return await run("fix", req, req.error_message, request)

    return router


code_router = make_cell_router("python")
sql_router = make_cell_router("sql")
markdown_router = make_cell_router("markdown")
