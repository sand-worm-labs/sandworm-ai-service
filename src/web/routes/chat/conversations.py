from fastapi import APIRouter

from . import store
from .models import Conversation, CreateConversationRequest, UpdateConversationRequest

router = APIRouter(prefix="/conversations")


@router.get("", response_model=list[Conversation])
async def list_conversations():
    return [Conversation(**c) for c in store.all_conversations()]


@router.post("", response_model=Conversation, status_code=201)
async def create_conversation(req: CreateConversationRequest):
    return Conversation(**store.create(req.title))


@router.get("/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    return Conversation(**store.get_or_404(conversation_id))


@router.patch("/{conversation_id}", response_model=Conversation)
async def update_conversation(conversation_id: str, req: UpdateConversationRequest):
    conv = store.get_or_404(conversation_id)
    conv["title"] = req.title
    conv["updated_at"] = store.now()
    return Conversation(**conv)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str):
    store.delete(conversation_id)
