import os

import litellm
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from . import store
from .completions import _OPENROUTER_BASE, _openrouter_model, stream_chunks
from .models import SendMessageRequest

router = APIRouter(prefix="/conversations")


@router.post("/{conversation_id}/messages")
async def send_message(conversation_id: str, req: SendMessageRequest):
    conv = store.get_or_404(conversation_id)

    conv["messages"].append({"role": "user", "content": req.content})

    if req.stream:
        collected: list[str] = []

        async def _stream_and_save():
            async for chunk in stream_chunks(conv["messages"], req.model, req.temperature, req.max_tokens):
                collected.append(chunk)
                yield chunk
            conv["messages"].append({"role": "assistant", "content": "".join(collected)})
            conv["updated_at"] = store.now()

        return StreamingResponse(_stream_and_save(), media_type="text/plain")

    response = await litellm.acompletion(
        model=_openrouter_model(req.model),
        messages=conv["messages"],
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        api_base=_OPENROUTER_BASE,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    assistant_content = response.choices[0].message.content
    conv["messages"].append({"role": "assistant", "content": assistant_content})
    conv["updated_at"] = store.now()

    return {
        "message": {"role": "assistant", "content": assistant_content},
        "model": response.model,
        "usage": dict(response.usage) if response.usage else None,
    }
