from fastapi import APIRouter
from src.web.routes.sql.models import EditSqlRequest, FixSqlRequest, SqlResponse
from src.services.sql.service import SqlService

router = APIRouter()


@router.post("/edit", response_model=SqlResponse)
async def edit_sql_route(req: EditSqlRequest):
    service = SqlService(api_key=req.openrouter_api_key, model=req.model)
    code = await service.edit(req)
    return SqlResponse(code=code, context=req.context)


@router.post("/fix", response_model=SqlResponse)
async def fix_sql_route(req: FixSqlRequest):
    service = SqlService(api_key=req.openrouter_api_key, model=req.model)
    code = await service.fix(req)
    return SqlResponse(code=code, context=req.context)
