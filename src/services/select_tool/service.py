from src.services.select_tool.models import SelectToolRequest, SelectToolResponse
from src.services.sandworm_tools.service import SandwormToolsService


class SelectToolService:
    def __init__(self, req: SelectToolRequest, embed_fn):
        self.req = req
        self._tools = SandwormToolsService(embed_fn)

    async def select(self) -> list[SelectToolResponse]:
        user_message = next(m for m in reversed(self.req.messages) if m.role == "user")

        results = await self._tools.search(query=user_message.content, top_k=5)
        if not results:
            raise ValueError("No matching tool found")

        return [
            SelectToolResponse(
                tool_id=tool["tool_id"],
                tool_name=tool.get("g3") or tool["tool_id"],
                category=tool.get("g1") or "",
                viz=tool.get("viz"),
                inputs=tool.get("inputs", []),
                returns=tool.get("returns", []),
                score=0.0,
            )
            for tool in results
        ]
