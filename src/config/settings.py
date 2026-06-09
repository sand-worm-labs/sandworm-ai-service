from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    AI_HANDSHAKE_TOKEN: str
    AI_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    NEST_BASE_URL: str = "http://localhost:8081/api/"
    REDIS_URL: str = "redis://host.docker.internal:6379/0"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent / ".env",
    )

    @property
    def handshake_token(self) -> str:
        return self.AI_HANDSHAKE_TOKEN

    @property
    def openrouter_base_url(self) -> str:
        return self.AI_OPENROUTER_BASE_URL

    @property
    def nest_base_url(self) -> str:
        return self.NEST_BASE_URL

    @property
    def redis_url(self) -> str:
        return self.REDIS_URL

settings = Settings()