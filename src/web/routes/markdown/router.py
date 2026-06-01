from fastapi import APIRouter
from src.web.routes.markdown.models import EditMarkdownRequest, MarkdownResponse
from src.services.markdown.service import MarkdownService

router = APIRouter()


@router.post("/edit", response_model=MarkdownResponse)
async def edit_markdown_route(req: EditMarkdownRequest):
    service = MarkdownService(api_key=req.openrouter_api_key, model=req.model)
    content = await service.edit(req)
    return MarkdownResponse(content=content, context=req.context)
