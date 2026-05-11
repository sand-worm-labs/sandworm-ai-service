from fastapi import APIRouter

from .completions import router as completions_router
from .conversations import router as conversations_router
from .messages import router as messages_router

router = APIRouter(prefix="/chat", tags=["chat"])

router.include_router(completions_router)
router.include_router(conversations_router)
router.include_router(messages_router)
