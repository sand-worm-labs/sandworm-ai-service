from fastapi import APIRouter
from src.web.routes.document.models import DocumentTitleRequest, DocumentTitleResponse
from src.services.document_title import generate_document_title

router = APIRouter()


@router.post("/generate-title", response_model=DocumentTitleResponse)
async def document_title_route(req: DocumentTitleRequest):
    return await generate_document_title(req)
