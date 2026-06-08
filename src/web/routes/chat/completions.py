from dataclasses import asdict
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from src.services.pipeline.service import PipelineState, run_pipeline
from src.services.completions.models import CompletionRequest

router = APIRouter()


@router.post("/completions")
async def completions_route(req: CompletionRequest) -> JSONResponse:
    state = PipelineState(
        messages=req.messages,
        model=req.model,
        api_key=req.openrouter_api_key,
        context=req.context,
    )
    result = await run_pipeline(state)
    return JSONResponse(content={
        "parsed_intent": asdict(result.parsed_intent) if result.parsed_intent else None,
        "block_plan": result.block_plan.model_dump() if result.block_plan else None,
        "generated_blocks": [b.model_dump() for b in result.generated_blocks] if result.generated_blocks else None,
        "notebook_markdown": result.notebook_markdown,
        "output": result.output,
    })