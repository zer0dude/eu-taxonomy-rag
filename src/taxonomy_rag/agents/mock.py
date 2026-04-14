"""Mock agent for evaluation scaffolding.

Returns a fixed hardcoded answer regardless of any input.
Useful for testing the eval pipeline end-to-end before real agents exist.

AgentProtocol has moved to agents/base.py.
"""

from __future__ import annotations

from taxonomy_rag.readers.base import AttachmentInfo  # noqa: F401 (re-exported for convenience)
from taxonomy_rag.tracing.base import NullTracer


def get_agent() -> "MockAgent":
    return MockAgent()


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

    def answer(
        self,
        question: str,
        context: str = "",
        prompt: str = "",
        attachments: list[AttachmentInfo] = [],
        tracer: NullTracer = NullTracer(),
    ) -> str:  # noqa: ARG002
        return self.ANSWER
