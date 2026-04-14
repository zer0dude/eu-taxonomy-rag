"""Tracing primitives — LogLevel, Tracer Protocol, and NullTracer.

Intentionally import-free (no project imports) so this module can be
imported anywhere without pulling in heavy dependencies.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable


class LogLevel(str, Enum):
    FULL      = "full"       # store complete tool result text
    TRUNCATED = "truncated"  # store first N chars + total char count
    METADATA  = "metadata"   # store only char count and timing, no text
    NONE      = "none"       # NullTracer; no file written


@runtime_checkable
class Tracer(Protocol):
    """Interface that tracing implementations must satisfy."""

    def record_input(
        self,
        question: str,
        context: str,
        attachments: list[Any],
    ) -> None:
        """Record the inputs the agent received for this question."""
        ...

    def log_reasoning(self, iteration_index: int, text: str) -> None:
        """Record the LLM's reasoning text for a given iteration."""
        ...

    def log_tool_call(
        self,
        iteration_index: int,
        tool_name: str,
        tool_input: dict,
        result: str,
        duration_ms: float,
    ) -> None:
        """Record a single tool call and its result."""
        ...

    def record_output(self, final_answer: str, duration_seconds: float) -> None:
        """Record the agent's final answer and total wall-clock duration."""
        ...

    def save(self, path: Any) -> None:
        """Persist the accumulated trace to disk at the given path."""
        ...


class NullTracer:
    """No-op tracer.

    Used as the default everywhere so agents never need to guard with
    `if tracer is not None`. Stateless — safe as a default argument.
    """

    def record_input(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_reasoning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_tool_call(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_output(self, *args: Any, **kwargs: Any) -> None:
        pass

    def save(self, *args: Any, **kwargs: Any) -> None:
        pass
