from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.services.completions.models import CompletionRequest

router = APIRouter()


async def _stream_text(text: str):
    yield text


@router.post("/completions")
async def completions_route(req: CompletionRequest):
    return StreamingResponse(_stream_text("hello from completions"), media_type="text/event-stream")
