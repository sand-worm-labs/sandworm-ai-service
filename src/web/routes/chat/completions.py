from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

from src.services.agent.service import AgentState, run_chat
from src.services.completions.models import CompletionRequest
from src.util.cache import get_active_job, request_job_cancel, set_active_job

router = APIRouter()


@router.post("/completions")
async def completions_route(req: CompletionRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    state = AgentState(
        messages=req.messages,
        model=req.model,
        api_key=req.openrouter_api_key,
        context=req.context,
        **({"job_id": req.job_id} if req.job_id else {}),
    )
    # Every message gets its own job. One still running for an earlier message
    # is told to stop, so two never write to the notebook for long, and this
    # message is never left unanswered behind it.
    previous = await get_active_job(req.context.chat_id)
    if previous and previous != state.job_id:
        await request_job_cancel(previous)
    await set_active_job(req.context.chat_id, state.job_id)
    background_tasks.add_task(run_chat, state)
    return JSONResponse(content={"job_id": state.job_id, "active": False})
