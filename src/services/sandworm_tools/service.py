from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.config.settings import settings
from src.services.sandworm_tools.models import SandwormTool

COLLECTION = "sandworm_tools"
VECTOR_SIZE = 1536


def _client() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


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
    def __init__(self, embed_fn):
        """embed_fn: async callable (text: str) -> list[float]"""
        self._embed = embed_fn
        self._client = _client()

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(COLLECTION)
        if not exists:
            await self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    async def upsert(self, tools: list[SandwormTool]) -> None:
        await self.ensure_collection()
        points = []
        for tool in tools:
            text = _build_embedding_text(tool)
            vector = await self._embed(text)
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
        vector = await self._embed(query)

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
