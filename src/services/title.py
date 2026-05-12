from src.web.routes.chat.models import TitleRequest, TitleResponse

async def generate_title(req: TitleRequest) -> TitleResponse:
    return TitleResponse(
        title="Sample Chat Title",
        context=req.context,
    )