from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.services.pipeline.service import PipelineState, run_pipeline
from src.services.completions.service import CompletionService
from src.services.completions.models import CompletionRequest

router = APIRouter()
service = CompletionService()

@router.post("/completions")
async def completions_route(req: CompletionRequest) -> StreamingResponse:
    state = PipelineState(
            messages=req.messages,
            model=req.model,
            api_key=req.openrouter_api_key,
            context=req.context,
        )
    result = await run_pipeline(state)

    async def _stream():
        yield result.output

    return StreamingResponse(_stream(), media_type="text/event-stream")