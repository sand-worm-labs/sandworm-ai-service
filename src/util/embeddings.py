from __future__ import annotations

import asyncio

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


# Free, local embeddings for notebook memory: no API key, no per-call cost.
# fastembed runs a small ONNX model inside this service; the model file is
# downloaded once on first use and cached.
LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_DIMENSIONS = 384
_local_model = None


def _load_local_model():
    global _local_model
    if _local_model is None:
        from fastembed import TextEmbedding

        _local_model = TextEmbedding(LOCAL_MODEL)
    return _local_model


def _embed_local_sync(texts: list[str], query: bool) -> list[list[float]]:
    model = _load_local_model()
    vectors = model.query_embed(texts) if query else model.embed(texts)
    return [v.tolist() for v in vectors]


async def embed_local(texts: list[str], query: bool = False) -> list[list[float]]:
    """Embed on this machine. query=True for a search question, False for stored text."""
    return await asyncio.to_thread(_embed_local_sync, texts, query)
