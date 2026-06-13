from fastapi import APIRouter
from src.web.routes.select_tool.models import SelectToolRequest, SelectToolResponse
from src.services.select_tool.service import SelectToolService

router = APIRouter()


@router.post("/select", response_model=list[SelectToolResponse])
async def select_tool_route(req: SelectToolRequest):
    return await SelectToolService(req).select()
