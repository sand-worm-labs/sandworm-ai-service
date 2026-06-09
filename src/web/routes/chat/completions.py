from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from src.services.pipeline.service import PipelineState, run_pipeline
from src.services.completions.models import CompletionRequest
from src.util.cache import get_active_job, set_active_job

router = APIRouter()


@router.post("/completions")
async def completions_route(req: CompletionRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    existing_job_id = await get_active_job(req.context.chat_id)
    if existing_job_id:
        return JSONResponse(content={"job_id": existing_job_id, "active": True})

    state = PipelineState(
        messages=req.messages,
        model=req.model,
        api_key=req.openrouter_api_key,
        context=req.context,
    )
    await set_active_job(req.context.chat_id, state.job_id)
    background_tasks.add_task(run_pipeline, state)
    return JSONResponse(content={"job_id": state.job_id, "active": False})