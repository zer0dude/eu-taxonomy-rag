"""LLM provider helpers for LiteLLM.

Public API:
    get_model_string(provider)     → LiteLLM model string
    get_completion_kwargs(provider) → full kwargs dict for litellm.completion()

Usage pattern::

    import litellm
    from taxonomy_rag.llm.provider import get_completion_kwargs

    response = litellm.completion(**get_completion_kwargs(), messages=messages)

Provider strings: "ollama" | "anthropic" | "openai" | None (uses settings default).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from taxonomy_rag.config import settings

SUPPORTED_PROVIDERS = ("ollama", "anthropic", "openai")


def get_model_string(provider: str | None = None) -> str:
    """Return the LiteLLM model string for the given provider.

    When provider is None, delegates to settings.litellm_model_string which
    reads the LLM_PROVIDER env var.  Pass an explicit provider to override
    on a per-call basis (used by RAG query() methods).
    """
    if provider is None:
        return settings.litellm_model_string

    if provider == "ollama":
        return f"ollama/{settings.ollama_model}"
    if provider == "anthropic":
        return settings.anthropic_model
    if provider == "openai":
        return settings.openai_model

    raise ValueError(
        f"Unknown provider: {provider!r}. "
        f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}"
    )


@lru_cache(maxsize=4)
def get_completion_kwargs(provider: str | None = None) -> dict[str, Any]:
    """Return a kwargs dict ready for ``litellm.completion(**kwargs, messages=...)``.

    Always includes ``"model"``.  Adds provider-specific extras:
    - Ollama: ``api_base`` pointing at the local Ollama server
    - Anthropic/OpenAI: ``api_key`` when set in config (LiteLLM also reads env vars)

    Cached per provider string so settings are only read once per process.
    Call ``get_completion_kwargs.cache_clear()`` in tests that patch settings.
    """
    p = provider or settings.llm_provider
    kwargs: dict[str, Any] = {"model": get_model_string(provider)}

    if p == "ollama":
        kwargs["api_base"] = settings.ollama_base_url

    if p == "anthropic" and settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key

    if p == "openai" and settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key

    return kwargs
