from fastapi import APIRouter
from src.web.routes.chat.models import CompletionRequest, CompletionResponse
from src.services.completion import complete

router = APIRouter()

@router.post("/completions", response_model=CompletionResponse)
async def completions_route(req: CompletionRequest):
    return await complete(req)