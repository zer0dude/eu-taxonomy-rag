"""Unit tests for AgentLoop — all litellm.completion() calls are mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from taxonomy_rag.llm.loop import AgentLoop
from taxonomy_rag.tracing.base import NullTracer


# ---------------------------------------------------------------------------
# Helpers to build mock litellm responses
# ---------------------------------------------------------------------------

def _text_response(content: str) -> MagicMock:
    """Simulate a model response that returns plain text (no tool calls)."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    msg.model_dump.return_value = {"role": "assistant", "content": content, "tool_calls": None}

    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _tool_call_response(
    tool_name: str,
    arguments: dict,
    call_id: str = "call_abc123",
    reasoning: str = "",
) -> MagicMock:
    """Simulate a model response that calls one tool."""
    func = MagicMock()
    func.name = tool_name
    func.arguments = json.dumps(arguments)

    call = MagicMock()
    call.id = call_id
    call.function = func

    msg = MagicMock()
    msg.content = reasoning  # often None/empty when calling tools
    msg.tool_calls = [call]
    msg.model_dump.return_value = {
        "role": "assistant",
        "content": reasoning,
        "tool_calls": [
            {"id": call_id, "function": {"name": tool_name, "arguments": json.dumps(arguments)}}
        ],
    }

    choice = MagicMock()
    choice.finish_reason = "tool_calls"
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Minimal ToolKit stand-in (avoids real Tool protocol machinery)
# ---------------------------------------------------------------------------

class _FakeToolKit:
    def __init__(self, tool_result: str = "tool_output"):
        self._result = tool_result
        self.calls: list[tuple[str, dict]] = []

    def to_litellm_schema(self) -> list[dict]:
        return [{"type": "function", "function": {"name": "fake_tool", "description": "", "parameters": {}}}]

    def run(self, tool_name: str, tool_input: dict) -> str:
        self.calls.append((tool_name, tool_input))
        return self._result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentLoopTextResponse:
    def test_returns_text_when_model_stops_immediately(self):
        loop = AgentLoop(completion_kwargs={"model": "test-model"})
        toolkit = _FakeToolKit()

        with patch("taxonomy_rag.llm.loop.litellm.completion", return_value=_text_response("Hello back")):
            result = loop.run([{"role": "user", "content": "Hi"}], toolkit)

        assert result == "Hello back"
        assert toolkit.calls == []

    def test_tracer_log_reasoning_called_once(self):
        loop = AgentLoop(completion_kwargs={"model": "test-model"})
        toolkit = _FakeToolKit()
        tracer = MagicMock(spec=NullTracer)

        with patch("taxonomy_rag.llm.loop.litellm.completion", return_value=_text_response("Answer")):
            loop.run([{"role": "user", "content": "Q"}], toolkit, tracer=tracer)

        tracer.log_reasoning.assert_called_once_with(1, "Answer")

    def test_caller_message_list_not_mutated(self):
        loop = AgentLoop(completion_kwargs={"model": "test-model"})
        toolkit = _FakeToolKit()
        original = [{"role": "user", "content": "Q"}]
        msgs_copy = list(original)

        with patch("taxonomy_rag.llm.loop.litellm.completion", return_value=_text_response("A")):
            loop.run(original, toolkit)

        assert original == msgs_copy


class TestAgentLoopToolCall:
    def test_tool_called_once_then_text_returned(self):
        loop = AgentLoop(completion_kwargs={"model": "test-model"})
        toolkit = _FakeToolKit(tool_result="found_it")
        responses = [
            _tool_call_response("fake_tool", {"value": "test"}, "call_1"),
            _text_response("Final answer"),
        ]

        with patch("taxonomy_rag.llm.loop.litellm.completion", side_effect=responses):
            result = loop.run([{"role": "user", "content": "Q"}], toolkit)

        assert result == "Final answer"
        assert toolkit.calls == [("fake_tool", {"value": "test"})]

    def test_tool_result_appended_as_tool_role_message(self):
        loop = AgentLoop(completion_kwargs={"model": "test-model"})
        toolkit = _FakeToolKit(tool_result="result_string")
        captured: list[list[dict]] = []

        def capture_completion(**kwargs):
            captured.append(list(kwargs["messages"]))
            if len(captured) == 1:
                return _tool_call_response("fake_tool", {"value": "x"}, "id_123")
            return _text_response("Done")

        with patch("taxonomy_rag.llm.loop.litellm.completion", side_effect=capture_completion):
            loop.run([{"role": "user", "content": "Q"}], toolkit)

        # On the second call, messages should include the tool result
        second_msgs = captured[1]
        tool_msg = next((m for m in second_msgs if m.get("role") == "tool"), None)
        assert tool_msg is not None, "No tool-role message found in second call"
        assert tool_msg["tool_call_id"] == "id_123"
        assert tool_msg["content"] == "result_string"

    def test_tracer_log_tool_call_called_with_correct_args(self):
        loop = AgentLoop(completion_kwargs={"model": "test-model"})
        toolkit = _FakeToolKit(tool_result="result")
        tracer = MagicMock(spec=NullTracer)
        responses = [
            _tool_call_response("fake_tool", {"value": "v"}, "call_99"),
            _text_response("Answer"),
        ]

        with patch("taxonomy_rag.llm.loop.litellm.completion", side_effect=responses):
            loop.run([{"role": "user", "content": "Q"}], toolkit, tracer=tracer)

        tracer.log_tool_call.assert_called_once()
        args = tracer.log_tool_call.call_args[0]
        assert args[1] == "fake_tool"           # tool_name
        assert args[2] == {"value": "v"}         # tool_input dict
        assert args[3] == "result"               # result string
        assert isinstance(args[4], float)        # duration_ms

    def test_none_content_does_not_crash_when_tool_calling(self):
        """message.content is often None when finish_reason == tool_calls."""
        loop = AgentLoop(completion_kwargs={"model": "test-model"})
        toolkit = _FakeToolKit()
        # Explicitly pass None content
        resp1 = _tool_call_response("fake_tool", {}, reasoning="")
        resp1.choices[0].message.content = None
        responses = [resp1, _text_response("Done")]

        with patch("taxonomy_rag.llm.loop.litellm.completion", side_effect=responses):
            result = loop.run([{"role": "user", "content": "Q"}], toolkit)

        assert result == "Done"

    def test_multiple_tool_calls_in_one_turn(self):
        """Model returns two tool_calls in a single turn."""
        loop = AgentLoop(completion_kwargs={"model": "test-model"})
        toolkit = _FakeToolKit(tool_result="ok")

        # Build a response with two tool calls
        def _make_call(name, args, cid):
            c = MagicMock()
            c.id = cid
            c.function = MagicMock()
            c.function.name = name
            c.function.arguments = json.dumps(args)
            return c

        msg = MagicMock()
        msg.content = None
        msg.tool_calls = [
            _make_call("fake_tool", {"value": "a"}, "c1"),
            _make_call("fake_tool", {"value": "b"}, "c2"),
        ]
        msg.model_dump.return_value = {"role": "assistant", "content": None}
        choice = MagicMock()
        choice.finish_reason = "tool_calls"
        choice.message = msg
        multi_resp = MagicMock()
        multi_resp.choices = [choice]

        responses = [multi_resp, _text_response("Done")]

        with patch("taxonomy_rag.llm.loop.litellm.completion", side_effect=responses):
            loop.run([{"role": "user", "content": "Q"}], toolkit)

        assert len(toolkit.calls) == 2
        assert toolkit.calls[0] == ("fake_tool", {"value": "a"})
        assert toolkit.calls[1] == ("fake_tool", {"value": "b"})


class TestAgentLoopMaxIterations:
    def test_returns_sentinel_at_max_iterations(self):
        loop = AgentLoop(completion_kwargs={"model": "test-model"}, max_iterations=2)
        toolkit = _FakeToolKit()

        with patch(
            "taxonomy_rag.llm.loop.litellm.completion",
            return_value=_tool_call_response("fake_tool", {}),
        ):
            result = loop.run([{"role": "user", "content": "Q"}], toolkit)

        assert "Max iterations" in result

    def test_tool_called_exactly_max_iterations_times(self):
        loop = AgentLoop(completion_kwargs={"model": "test-model"}, max_iterations=3)
        toolkit = _FakeToolKit()

        with patch(
            "taxonomy_rag.llm.loop.litellm.completion",
            return_value=_tool_call_response("fake_tool", {}),
        ):
            loop.run([{"role": "user", "content": "Q"}], toolkit)

        assert len(toolkit.calls) == 3


class TestAgentLoopUnexpectedFinishReason:
    def test_returns_content_on_unknown_finish_reason(self):
        loop = AgentLoop(completion_kwargs={"model": "test-model"})
        toolkit = _FakeToolKit()

        msg = MagicMock()
        msg.content = "Partial content"
        msg.tool_calls = None
        choice = MagicMock()
        choice.finish_reason = "length"
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]

        with patch("taxonomy_rag.llm.loop.litellm.completion", return_value=resp):
            result = loop.run([{"role": "user", "content": "Q"}], toolkit)

        assert result == "Partial content"
