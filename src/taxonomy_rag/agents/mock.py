"""Mock agent for evaluation scaffolding.

Returns a fixed hardcoded answer regardless of any input.
Useful for testing the eval pipeline end-to-end before real agents exist.

Also defines AgentProtocol — the interface all agents must satisfy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentProtocol(Protocol):
    """Interface that all eval-compatible agents must satisfy."""

    def answer(self, question: str, context: str = "", prompt: str = "") -> str:
        """Return an answer string for the given question.

        Args:
            question: The question to answer.
            context: Optional inline context from the question file (e.g. scenario description).
            prompt: The system prompt loaded from a prompt file.
        """
        ...


class MockAgent:
    """Hardcoded-output agent for pipeline testing.

    Ignores all inputs. Use this to verify that the eval harness
    correctly records, formats, and persists outputs before wiring up
    a real agent.
    """

    ANSWER = (
        "[MOCK] This agent returns a fixed response regardless of input. "
        "Replace with a real agent to get meaningful answers."
    )

    def answer(self, question: str, context: str = "", prompt: str = "") -> str:  # noqa: ARG002
        return self.ANSWER
