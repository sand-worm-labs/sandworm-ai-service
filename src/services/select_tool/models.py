from pydantic import BaseModel


class SelectToolRequest(BaseModel):
    query: str
    top_k: int = 5
