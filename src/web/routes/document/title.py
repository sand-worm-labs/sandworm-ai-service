from fastapi import APIRouter
from src.web.routes.document.models import DocumentTitleRequest, DocumentTitleResponse
from src.services.title.service import DocumentTitleService

router = APIRouter()


@router.post("/generate-title", response_model=DocumentTitleResponse)
async def document_title_route(req: DocumentTitleRequest):
    title = await DocumentTitleService(req).generate()
    return DocumentTitleResponse(title=title, context=req.context)
