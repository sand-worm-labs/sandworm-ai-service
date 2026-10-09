from __future__ import annotations

import httpx

from src.config.settings import settings

MODEL = "openai/text-embedding-3-large"


async def embed_texts(texts: list[str], api_key: str | None = None) -> list[list[float]]:
    """Embed through OpenRouter. api_key is the requesting user's own key; None
    uses the system's embedding key (for background work with no user attached)."""
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{settings.openrouter_base_url.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {api_key or settings.OPENROUTER_EMBEDDING_KEY}"},
            json={"model": MODEL, "input": texts},
            timeout=60,
        )
        res.raise_for_status()
        return [item["embedding"] for item in res.json()["data"]]
