from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.services.completions.service import CompletionService
from src.services.completions.models import CompletionRequest

router = APIRouter()
service = CompletionService()

@router.post("/completions")
async def completions_route(req: CompletionRequest) -> StreamingResponse:
    
    return StreamingResponse(service.stream(req), media_type="text/event-stream")