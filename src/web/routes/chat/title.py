from fastapi import APIRouter
from src.web.routes.chat.models import TitleRequest, TitleResponse
from services.title.service import generate_title

router = APIRouter()

@router.post("/title", response_model=TitleResponse)
async def title_route(req: TitleRequest):
    return await generate_title(req)