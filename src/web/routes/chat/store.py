import uuid

from fastapi import HTTPException

from src.util.date import now

_conversations: dict[str, dict] = {}


def get_or_404(conversation_id: str) -> dict:
    conv = _conversations.get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


def all_conversations() -> list[dict]:
    return sorted(_conversations.values(), key=lambda c: c["updated_at"], reverse=True)


def create(title: str) -> dict:
    n = now()
    conv = {"id": str(uuid.uuid4()), "title": title, "created_at": n, "updated_at": n, "messages": []}
    _conversations[conv["id"]] = conv
    return conv


def delete(conversation_id: str) -> None:
    get_or_404(conversation_id)
    del _conversations[conversation_id]
