from fastapi import APIRouter
from src.services.chat_title.service import ChatTitleService
from src.services.completions.models import CompletionRequest

router = APIRouter()

@router.post("/title")
async def title_route(req: CompletionRequest):
    title = await ChatTitleService(req).generate()
    return {"title": title}
