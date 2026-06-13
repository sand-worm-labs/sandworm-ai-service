from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    AI_HANDSHAKE_TOKEN: str
    AI_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    NEST_BASE_URL: str = "http://localhost:8081/api/"
    REDIS_URL: str = "redis://host.docker.internal:6379/0"
    QDRANT_URL: str = "http://host.docker.internal:6333"
    QDRANT_API_KEY: str | None = None
    OPENROUTER_API_KEY: str 

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

    @property
    def qdrant_url(self) -> str:
        return self.QDRANT_URL

    @property
    def qdrant_api_key(self) -> str | None:
        return self.QDRANT_API_KEY

    @property
    def openrouter_api_key(self) -> str:
        return self.OPENROUTER_API_KEY

settings = Settings()