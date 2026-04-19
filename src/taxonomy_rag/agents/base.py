"""AgentProtocol — the interface all eval-compatible agents must satisfy.

Kept in its own module so it can be imported by any agent without pulling in
agent-specific dependencies (Anthropic SDK, LangChain, etc.).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from taxonomy_rag.readers.base import AttachmentInfo
from taxonomy_rag.tracing.base import NullTracer, Tracer


@runtime_checkable
class AgentProtocol(Protocol):
    """Interface that all eval-compatible agents must satisfy.

    Every agent module must also expose a module-level get_agent() factory
    function so evaluate.py can load agents by name without hardcoding imports.
    """

    def answer(
        self,
        question: str,
        context: str = "",
        prompt: str = "",
        attachments: list[AttachmentInfo] | None = None,
        tracer: Tracer = NullTracer(),
    ) -> str:
        """Return an answer string for the given question.

        Args:
            question:    The question to answer.
            context:     Optional inline scenario text from the question file.
            prompt:      System prompt loaded from a prompt file.
            attachments: Metadata list of available attached files. Agents
                         that use tools can read these via tool calls; agents
                         that don't (mock, llm_direct) may ignore this field.
            tracer:      Tracing sink. Pass a FileTracer to persist the agent's
                         reasoning loop; NullTracer (default) discards all calls.
        """
        ...
