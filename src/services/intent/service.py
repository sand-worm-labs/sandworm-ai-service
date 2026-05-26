import json
import litellm
from src.config.settings import settings
from .models import Intent, ParseIntentRequest, ParseIntentResponse

_SYSTEM_PROMPT = """\
You are an intent parser for a document-editor AI assistant.

Given a user message, extract the intent and return ONLY valid JSON — no explanation, \
no markdown fences — with exactly these three fields:

{
  "goal":   "<action the user wants>",
  "entity": "<document fragment being acted on>",
  "params": { "<key>": "<value>", ... }
}

goal   — one of: summarize, rewrite, translate, expand, shorten, analyze, explain,
         extract, format, generate, compare, search, edit, fix.
         Use the closest match; fall back to a short verb if none fits.

entity — one of: document, paragraph, selection, heading, table, code, list,
         sentence, title.
         Use "document" when the whole document is implied.

params — any additional constraints mentioned by the user, e.g.
         {"language": "French"}, {"tone": "formal"}, {"length": "short"},
         {"format": "bullet_points"}.
         Omit (or use {}) if none are present.\
"""


async def parse_intent(req: ParseIntentRequest) -> ParseIntentResponse:
    """Call the LLM to extract a structured intent from *req.message*."""
    response = await litellm.acompletion(
        model=req.model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": req.message},
        ],
        api_key=req.openrouter_api_key,
        api_base=settings.openrouter_base_url,
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=256,
    )

    raw: str = response.choices[0].message.content or "{}"
    data: dict = json.loads(raw)

    intent = Intent(
        goal=data.get("goal", "analyze"),
        entity=data.get("entity", "document"),
        params=data.get("params", {}),
    )

    return ParseIntentResponse(intent=intent, context=req.context)
