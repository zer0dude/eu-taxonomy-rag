"""Unit tests for ToolKit — schema format and dispatch."""

from __future__ import annotations

import pytest

from taxonomy_rag.tools.base import ToolKit


class DoubleValueTool:
    name = "double_value"
    description = "Returns the input integer doubled as a string."
    input_schema = {
        "type": "object",
        "properties": {"n": {"type": "integer", "description": "The number to double."}},
        "required": ["n"],
    }

    def run(self, n: int) -> str:
        return str(n * 2)


class ExplodingTool:
    name = "exploding"
    description = "Always raises RuntimeError."
    input_schema = {"type": "object", "properties": {}, "required": []}

    def run(self) -> str:
        raise RuntimeError("boom")


class TestToLiteLLMSchema:
    def test_output_has_openai_function_format(self):
        tk = ToolKit([DoubleValueTool()])
        schema = tk.to_litellm_schema()
        assert len(schema) == 1
        assert schema[0]["type"] == "function"
        assert "function" in schema[0]

    def test_function_dict_has_name_description_parameters(self):
        tk = ToolKit([DoubleValueTool()])
        fn = tk.to_litellm_schema()[0]["function"]
        assert fn["name"] == "double_value"
        assert fn["description"] == "Returns the input integer doubled as a string."
        assert fn["parameters"] == DoubleValueTool.input_schema

    def test_key_is_parameters_not_input_schema(self):
        tk = ToolKit([DoubleValueTool()])
        fn = tk.to_litellm_schema()[0]["function"]
        assert "parameters" in fn
        assert "input_schema" not in fn

    def test_multiple_tools_all_appear(self):
        tk = ToolKit([DoubleValueTool(), ExplodingTool()])
        schema = tk.to_litellm_schema()
        assert len(schema) == 2
        names = {s["function"]["name"] for s in schema}
        assert names == {"double_value", "exploding"}


class TestToolKitRun:
    def test_dispatches_to_correct_tool(self):
        tk = ToolKit([DoubleValueTool()])
        result = tk.run("double_value", {"n": 21})
        assert result == "42"

    def test_unknown_tool_returns_error_string_not_exception(self):
        tk = ToolKit([DoubleValueTool()])
        result = tk.run("nonexistent", {})
        assert "unknown tool" in result.lower()
        assert "nonexistent" in result

    def test_tool_exception_returns_error_string_not_exception(self):
        tk = ToolKit([ExplodingTool()])
        result = tk.run("exploding", {})
        assert "Error running tool" in result
        assert "boom" in result

    def test_tool_receives_kwargs_correctly(self):
        class RecordingTool:
            name = "recorder"
            description = "Records call args."
            input_schema = {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            }
            last_call: dict = {}

            def run(self, a: str, b: int) -> str:
                RecordingTool.last_call = {"a": a, "b": b}
                return "recorded"

        tool = RecordingTool()
        tk = ToolKit([tool])
        tk.run("recorder", {"a": "hello", "b": 42})
        assert RecordingTool.last_call == {"a": "hello", "b": 42}
