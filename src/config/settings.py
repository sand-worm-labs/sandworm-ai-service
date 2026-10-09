from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    AI_HANDSHAKE_TOKEN: str
    AI_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    REDIS_URL: str = "redis://host.docker.internal:6379/0"
    QDRANT_URL: str = "http://host.docker.internal:6333"
    QDRANT_API_KEY: str | None = None
    OPENROUTER_EMBEDDING_KEY: str
    # Chat is a tool-calling loop over the MCP server (the same agent external
    # clients use); every notebook read and write goes through it.
    MCP_URL: str = "http://host.docker.internal:6789/mcp"
    AGENT_MAX_STEPS: int = 15
    # Hard limits per chat run, on top of the step cap. Seconds of wall-clock
    # time (the user's access token lasts 15 minutes) and total model tokens.
    AGENT_MAX_SECONDS: int = 600
    AGENT_MAX_TOKENS: int = 300_000
    # Comma-separated MCP tool names. If INCLUDE is set only those are offered
    # to the model; EXCLUDE removes names. Useful to shrink the tool list for
    # weaker models.
    AGENT_INCLUDE_TOOLS: str = ""
    AGENT_EXCLUDE_TOOLS: str = ""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent / ".env",
        extra="ignore",
    )

    @property
    def handshake_token(self) -> str:
        return self.AI_HANDSHAKE_TOKEN

    @property
    def openrouter_base_url(self) -> str:
        return self.AI_OPENROUTER_BASE_URL

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
    def openrouter_embedding_key(self) -> str:
        return self.OPENROUTER_EMBEDDING_KEY

settings = Settings()