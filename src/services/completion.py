import random
from src.web.routes.chat.models import CompletionRequest, CompletionResponse, Message

_DUMB_RESPONSES = [
    "Have you tried turning it off and on again?",
    "Blockchain go brrr.",
    "I am just a simple AI, I know nothing.",
    "The answer is 42. It's always 42.",
    "I was going to help but I forgot how.",
    "Error 404: Intelligence not found.",
    "Interesting. Tell me more while I pretend to care.",
    "My other response is a Lamborghini.",
]

async def complete(req: CompletionRequest) -> CompletionResponse:
    return CompletionResponse(
        message=Message(
            role="assistant",
            content=random.choice(_DUMB_RESPONSES),
        ),
        model=req.model,
        context=req.context,
        finish_reason="stop",
    )
