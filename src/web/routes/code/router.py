from fastapi import APIRouter
from src.web.routes.code.models import EditCodeRequest, FixCodeRequest, CodeResponse
from src.services.code.service import CodeService

router = APIRouter()


@router.post("/edit", response_model=CodeResponse)
async def edit_code_route(req: EditCodeRequest):
    service = CodeService(api_key=req.openrouter_api_key, model=req.model)
    code = await service.edit(req)
    return CodeResponse(code=code, context=req.context)


@router.post("/fix", response_model=CodeResponse)
async def fix_code_route(req: FixCodeRequest):
    service = CodeService(api_key=req.openrouter_api_key, model=req.model)
    code = await service.fix(req)
    return CodeResponse(code=code, context=req.context)
