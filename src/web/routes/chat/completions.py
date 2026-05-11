from typing import AsyncIterator

import litellm
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .models import CompletionRequest, CompletionResponse, Message

router = APIRouter()

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _openrouter_model(model: str) -> str:
    return model if model.startswith("openrouter/") else f"openrouter/{model}"


async def stream_chunks(
    messages: list[dict],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int | None,
) -> AsyncIterator[str]:
    response = await litellm.acompletion(
        model=_openrouter_model(model),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_base=_OPENROUTER_BASE,
        api_key=api_key,
        stream=True,
    )
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


@router.post("/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest):
    """Stateless one-shot completion."""
    if req.stream:
        return StreamingResponse(
            stream_chunks([m.model_dump() for m in req.messages], req.model, req.openrouter_api_key, req.temperature, req.max_tokens),
            media_type="text/plain",
        )

    response = await litellm.acompletion(
        model=_openrouter_model(req.model),
        messages=[m.model_dump() for m in req.messages],
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        api_base=_OPENROUTER_BASE,
        api_key=req.openrouter_api_key,
    )
    choice = response.choices[0]
    return CompletionResponse(
        message=Message(role="assistant", content=choice.message.content),
        model=response.model,
        usage=dict(response.usage) if response.usage else None,
    )
