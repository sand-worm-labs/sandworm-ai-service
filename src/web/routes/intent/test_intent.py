from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.services.intent.models import ParseIntentRequest
from src.services.intent.service import ParseIntentService

router = APIRouter()

@router.post("/parse_intent")
async def parse_intent(req: ParseIntentRequest):
    service = ParseIntentService(req)
    return StreamingResponse(service.stream(), media_type="text/event-stream")