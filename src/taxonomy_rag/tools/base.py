"""Tool interface and ToolKit for Anthropic-compatible agent tool use.

Tool     — Protocol that every tool must satisfy. The name, description, and
           input_schema fields are passed directly to the Anthropic Messages API.

ToolKit  — A named collection of tools. Converts to Anthropic schema and
           dispatches tool_use blocks returned by the model.

Adding a new tool only requires implementing Tool and including an instance
in the ToolKit passed to the agent — no changes needed elsewhere.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Interface every tool must satisfy."""

    name: str
    """Unique snake_case identifier shown to the model."""

    description: str
    """Plain-English explanation of what the tool does and when to use it."""

    input_schema: dict
    """JSON Schema object describing the tool's parameters (Anthropic format)."""

    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a plain-text result."""
        ...


class ToolKit:
    """Groups tools and handles schema conversion and dispatch.

    Instantiate once per agent call (because tools may hold per-question state
    such as an attachment path map).
    """

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def to_anthropic_schema(self) -> list[dict]:
        """Return the tools list in the format expected by client.messages.create()."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def run(self, tool_name: str, tool_input: dict) -> str:
        """Dispatch a tool_use block: look up the tool by name and call run()."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return f"Error: unknown tool '{tool_name}'."
        try:
            return tool.run(**tool_input)
        except Exception as exc:
            return f"Error running tool '{tool_name}': {exc}"
