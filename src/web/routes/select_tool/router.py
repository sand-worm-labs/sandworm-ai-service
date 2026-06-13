from fastapi import APIRouter
from src.services.select_tool.models import SelectToolRequest
from src.services.sandworm_tools.models import SandwormTool
from src.services.sandworm_tools.service import SandwormToolsService

router = APIRouter()


@router.post("/search", response_model=list[SandwormTool])
async def search_tools(req: SelectToolRequest):
    service = SandwormToolsService()
    results = await service.search(query=req.query, top_k=req.top_k)
    return [SandwormTool(**r) for r in results]
