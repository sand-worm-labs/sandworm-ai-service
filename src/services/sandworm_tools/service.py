from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("sandworm.tools")

import httpx
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from src.config.settings import settings
from src.services.sandworm_tools.models import SandwormTool
from src.util.qdrant import get_qdrant

COLLECTION = "sandworm_tools"


def _build_embedding_text(tool: SandwormTool) -> str:
    tags = " > ".join(t for t in [tool.g1, tool.g2, tool.g3, tool.g4, tool.g5] if t)

    return_fields = [r.name for r in tool.returns]
    input_labels = [i.label for i in tool.inputs]

    parts = [
        tool.tool_id,
        tool.description,
        f"category: {tags}" if tags else "",
        f"outputs: {', '.join(return_fields)}" if return_fields else "",
        f"inputs: {', '.join(input_labels)}" if input_labels else "",
    ]
    return " | ".join(p for p in parts if p)


class SandwormToolsService:
    def __init__(self):
        self._client = get_qdrant()

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
                json={"model": "openai/text-embedding-3-large", "input": texts},
                timeout=60,
            )
            res.raise_for_status()
            return [item["embedding"] for item in res.json()["data"]]

    async def embed(self, text: str) -> list[float]:
        return (await self._embed_batch([text]))[0]

    async def upsert(self, tools: list[SandwormTool], batch_size: int = 400) -> None:
        for i in range(0, len(tools), batch_size):
            batch = tools[i : i + batch_size]
            log.info("batch %d — embedding %d tools (%d–%d of %d)", i // batch_size + 1, len(batch), i + 1, min(i + batch_size, len(tools)), len(tools))
            texts = [_build_embedding_text(t) for t in batch]
            vectors = await self._embed_batch(texts)
            points = []
            for tool, vector in zip(batch, vectors):
                points.append(
                    PointStruct(
                        id=abs(hash(tool.tool_id)) % (2**63),
                        vector=vector,
                        payload={
                            "tool_id": tool.tool_id,
                            "g1": tool.g1,
                            "g2": tool.g2,
                            "g3": tool.g3,
                            "g4": tool.g4,
                            "g5": tool.g5,
                            "description": tool.description,
                            "scope": tool.scope,
                            "returns": [r.model_dump() for r in tool.returns],
                            "inputs": [i.model_dump() for i in tool.inputs],
                        },
                    )
                )
            await self._client.upsert(collection_name=COLLECTION, points=points)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_g1: str | None = None,
    ) -> list[dict[str, Any]]:
        vector = await self.embed(query)

        qdrant_filter = None
        if filter_g1:
            qdrant_filter = Filter(
                must=[FieldCondition(key="g1", match=MatchValue(value=filter_g1))]
            )

        results = await self._client.query_points(
            collection_name=COLLECTION,
            query=vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [hit.payload for hit in results.points]
