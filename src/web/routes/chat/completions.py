import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.services.pipeline.service import PipelineState, run_pipeline
from src.services.completions.models import CompletionRequest

router = APIRouter()


@router.post("/completions")
async def completions_route(req: CompletionRequest) -> StreamingResponse:
    state = PipelineState(
        messages=req.messages,
        model=req.model,
        api_key=req.openrouter_api_key,
        context=req.context,
    )

    async def _stream():
        async for part in run_pipeline(state):
            yield json.dumps(part) + "\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")