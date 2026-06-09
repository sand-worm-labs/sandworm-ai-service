import asyncio
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from src.services.pipeline.service import PipelineState, run_pipeline
from src.services.completions.models import CompletionRequest

router = APIRouter()


@router.post("/completions")
async def completions_route(req: CompletionRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    state = PipelineState(
        messages=req.messages,
        model=req.model,
        api_key=req.openrouter_api_key,
        context=req.context,
    )
    background_tasks.add_task(run_pipeline, state)
    return JSONResponse(content={"job_id": state.job_id})