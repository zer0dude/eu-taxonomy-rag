from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from taxonomy_rag.config import settings

SUPPORTED_PROVIDERS = ("ollama", "anthropic", "openai")


@lru_cache(maxsize=4)
def get_llm(provider: str | None = None) -> BaseChatModel:
    """Return a LangChain chat model for the given provider.

    Defaults to settings.llm_provider when provider is None.
    lru_cache keeps one instance per provider string so models aren't
    re-initialised on every call.

    Imports are lazy: only the relevant langchain-* package is imported,
    so you only need the API key/service for the provider you actually use.
    """
    p = provider or settings.llm_provider

    if p == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )

    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )

    if p == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )

    raise ValueError(
        f"Unknown LLM provider: {p!r}. "
        f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}"
    )
