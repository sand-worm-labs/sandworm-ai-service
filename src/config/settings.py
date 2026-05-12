from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    AI_HANDSHAKE_TOKEN: str
    AI_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent / ".env",
    )

    @property
    def handshake_token(self) -> str:
        return self.AI_HANDSHAKE_TOKEN

    @property
    def openrouter_base_url(self) -> str:
        return self.AI_OPENROUTER_BASE_URL

settings = Settings()