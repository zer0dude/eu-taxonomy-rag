"""Unit tests for MockAgent and LLMDirectAgent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _litellm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestMockAgent:
    def test_returns_fixed_string(self):
        from taxonomy_rag.agents.mock import MockAgent
        agent = MockAgent()
        result = agent.answer("any question")
        assert "[MOCK]" in result

    def test_returns_same_answer_regardless_of_input(self):
        from taxonomy_rag.agents.mock import MockAgent
        agent = MockAgent()
        r1 = agent.answer("q1", context="ctx1", prompt="p1")
        r2 = agent.answer("q2", context="ctx2", prompt="p2")
        assert r1 == r2

    def test_answer_is_a_string(self):
        from taxonomy_rag.agents.mock import MockAgent
        agent = MockAgent()
        assert isinstance(agent.answer("q"), str)


class TestLLMDirectAgent:
    def test_returns_model_content(self):
        from taxonomy_rag.agents.llm_direct import LLMDirectAgent
        with patch("taxonomy_rag.agents.llm_direct.get_completion_kwargs", return_value={"model": "test"}):
            with patch("taxonomy_rag.agents.llm_direct.litellm.completion", return_value=_litellm_response("The direct answer.")):
                agent = LLMDirectAgent()
                result = agent.answer("What is taxonomy?")
        assert result == "The direct answer."

    def test_question_appears_in_user_message(self):
        from taxonomy_rag.agents.llm_direct import LLMDirectAgent
        captured: dict = {}

        def capture(**kwargs):
            captured["messages"] = kwargs["messages"]
            return _litellm_response("ok")

        with patch("taxonomy_rag.agents.llm_direct.get_completion_kwargs", return_value={"model": "test"}):
            with patch("taxonomy_rag.agents.llm_direct.litellm.completion", side_effect=capture):
                agent = LLMDirectAgent()
                agent.answer("What is taxonomy?")

        user_msg = captured["messages"][1]
        assert user_msg["role"] == "user"
        assert "What is taxonomy?" in user_msg["content"]

    def test_context_prepended_to_user_message(self):
        from taxonomy_rag.agents.llm_direct import LLMDirectAgent
        captured: dict = {}

        def capture(**kwargs):
            captured["messages"] = kwargs["messages"]
            return _litellm_response("ok")

        with patch("taxonomy_rag.agents.llm_direct.get_completion_kwargs", return_value={"model": "test"}):
            with patch("taxonomy_rag.agents.llm_direct.litellm.completion", side_effect=capture):
                agent = LLMDirectAgent()
                agent.answer("Question?", context="Important context.")

        user_content = captured["messages"][1]["content"]
        assert "Important context." in user_content
        assert "Question?" in user_content

    def test_system_message_uses_prompt(self):
        from taxonomy_rag.agents.llm_direct import LLMDirectAgent
        captured: dict = {}

        def capture(**kwargs):
            captured["messages"] = kwargs["messages"]
            return _litellm_response("ok")

        with patch("taxonomy_rag.agents.llm_direct.get_completion_kwargs", return_value={"model": "test"}):
            with patch("taxonomy_rag.agents.llm_direct.litellm.completion", side_effect=capture):
                agent = LLMDirectAgent()
                agent.answer("Q?", prompt="You are an EU Taxonomy expert.")

        sys_msg = captured["messages"][0]
        assert sys_msg["role"] == "system"
        assert "EU Taxonomy expert" in sys_msg["content"]

    def test_default_system_prompt_used_when_no_prompt(self):
        from taxonomy_rag.agents.llm_direct import LLMDirectAgent
        captured: dict = {}

        def capture(**kwargs):
            captured["messages"] = kwargs["messages"]
            return _litellm_response("ok")

        with patch("taxonomy_rag.agents.llm_direct.get_completion_kwargs", return_value={"model": "test"}):
            with patch("taxonomy_rag.agents.llm_direct.litellm.completion", side_effect=capture):
                agent = LLMDirectAgent()
                agent.answer("Q?")

        assert captured["messages"][0]["role"] == "system"
        assert captured["messages"][0]["content"]  # non-empty

    def test_none_content_returns_empty_string(self):
        from taxonomy_rag.agents.llm_direct import LLMDirectAgent
        msg = MagicMock()
        msg.content = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]

        with patch("taxonomy_rag.agents.llm_direct.get_completion_kwargs", return_value={"model": "test"}):
            with patch("taxonomy_rag.agents.llm_direct.litellm.completion", return_value=resp):
                agent = LLMDirectAgent()
                result = agent.answer("Q?")

        assert result == ""
