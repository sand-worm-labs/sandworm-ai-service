from __future__ import annotations

from src.services.agent.model import ask
from src.services.completions.models import CompletionRequest

from .prompts import SYSTEM_PROMPT


class ChatTitleService:
    def __init__(self, req: CompletionRequest) -> None:
        self.req = req

    async def generate(self) -> str:
        first_user = next((m for m in self.req.messages if m.role == "user"), None)
        if not first_user:
            return "New Chat"

        system = f"{self.req.derived_context}\n\n{SYSTEM_PROMPT}" if self.req.derived_context else SYSTEM_PROMPT
        return await ask(
            self.req.openrouter_api_key,
            self.req.model,
            system,
            first_user.content,
            self.req.temperature,
            self.req.max_tokens,
        )
