"""Unit tests for llm/provider.py — no network calls, settings are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_settings(**overrides):
    """Build a mock Settings object with sensible defaults."""
    s = MagicMock()
    s.llm_provider = overrides.get("llm_provider", "anthropic")
    s.ollama_model = overrides.get("ollama_model", "llama3.2")
    s.ollama_base_url = overrides.get("ollama_base_url", "http://localhost:11434")
    s.anthropic_model = overrides.get("anthropic_model", "claude-haiku-4-5-20251001")
    s.anthropic_api_key = overrides.get("anthropic_api_key", "sk-ant-test")
    s.openai_model = overrides.get("openai_model", "gpt-4o-mini")
    s.openai_api_key = overrides.get("openai_api_key", "sk-test")
    # litellm_model_string mirrors what config.py computes
    provider = s.llm_provider
    if provider == "ollama":
        s.litellm_model_string = f"ollama/{s.ollama_model}"
    elif provider == "anthropic":
        s.litellm_model_string = s.anthropic_model
    else:
        s.litellm_model_string = s.openai_model
    return s


class TestGetModelString:
    def test_ollama_returns_prefixed_model(self):
        from taxonomy_rag.llm.provider import get_model_string
        s = _make_settings(llm_provider="ollama", ollama_model="llama3.2")
        with patch("taxonomy_rag.llm.provider.settings", s):
            result = get_model_string("ollama")
        assert result == "ollama/llama3.2"

    def test_anthropic_returns_bare_model(self):
        from taxonomy_rag.llm.provider import get_model_string
        s = _make_settings(llm_provider="anthropic")
        with patch("taxonomy_rag.llm.provider.settings", s):
            result = get_model_string("anthropic")
        assert result == "claude-haiku-4-5-20251001"

    def test_openai_returns_bare_model(self):
        from taxonomy_rag.llm.provider import get_model_string
        s = _make_settings(llm_provider="openai", openai_model="gpt-4o-mini")
        with patch("taxonomy_rag.llm.provider.settings", s):
            result = get_model_string("openai")
        assert result == "gpt-4o-mini"

    def test_none_delegates_to_settings_litellm_model_string(self):
        from taxonomy_rag.llm.provider import get_model_string
        s = _make_settings(llm_provider="anthropic")
        with patch("taxonomy_rag.llm.provider.settings", s):
            result = get_model_string(None)
        assert result == s.litellm_model_string

    def test_unknown_provider_raises_value_error(self):
        from taxonomy_rag.llm.provider import get_model_string
        s = _make_settings()
        with patch("taxonomy_rag.llm.provider.settings", s):
            with pytest.raises(ValueError, match="Unknown provider"):
                get_model_string("groq")


class TestGetCompletionKwargs:
    def setup_method(self):
        from taxonomy_rag.llm.provider import get_completion_kwargs
        get_completion_kwargs.cache_clear()

    def test_ollama_includes_api_base(self):
        from taxonomy_rag.llm.provider import get_completion_kwargs
        s = _make_settings(llm_provider="ollama", ollama_base_url="http://localhost:11434")
        with patch("taxonomy_rag.llm.provider.settings", s):
            kwargs = get_completion_kwargs("ollama")
        assert kwargs["model"] == "ollama/llama3.2"
        assert kwargs["api_base"] == "http://localhost:11434"

    def test_anthropic_no_api_base(self):
        from taxonomy_rag.llm.provider import get_completion_kwargs
        s = _make_settings(llm_provider="anthropic")
        with patch("taxonomy_rag.llm.provider.settings", s):
            kwargs = get_completion_kwargs("anthropic")
        assert "api_base" not in kwargs
        assert kwargs["model"] == "claude-haiku-4-5-20251001"
        assert kwargs.get("api_key") == "sk-ant-test"

    def test_openai_no_api_base(self):
        from taxonomy_rag.llm.provider import get_completion_kwargs
        s = _make_settings(llm_provider="openai")
        with patch("taxonomy_rag.llm.provider.settings", s):
            kwargs = get_completion_kwargs("openai")
        assert "api_base" not in kwargs
        assert kwargs["model"] == "gpt-4o-mini"

    def test_anthropic_no_api_key_when_empty(self):
        from taxonomy_rag.llm.provider import get_completion_kwargs
        s = _make_settings(llm_provider="anthropic", anthropic_api_key="")
        with patch("taxonomy_rag.llm.provider.settings", s):
            kwargs = get_completion_kwargs("anthropic")
        assert "api_key" not in kwargs

    def test_cache_cleared_between_calls(self):
        """Verify the cache doesn't leak stale data between test runs."""
        from taxonomy_rag.llm.provider import get_completion_kwargs
        get_completion_kwargs.cache_clear()
        s1 = _make_settings(llm_provider="ollama")
        with patch("taxonomy_rag.llm.provider.settings", s1):
            k1 = get_completion_kwargs("ollama")
        assert k1["model"] == "ollama/llama3.2"
