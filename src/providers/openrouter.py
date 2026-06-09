from langchain_openrouter import ChatOpenRouter


def make_llm(
    api_key: str,
    model: str,
    temperature: float = 0,
    max_tokens: int | None = None,
) -> ChatOpenRouter:
    return ChatOpenRouter(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def make_streaming_llm(
    api_key: str,
    model: str,
    temperature: float = 0,
    max_tokens: int | None = None,
) -> ChatOpenRouter:
    return ChatOpenRouter(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
    )
