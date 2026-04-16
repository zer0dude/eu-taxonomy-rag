"""ReAct agent — LiteLLM tool-use loop with attachment reading.

The agent receives a list of attachment metadata (names, types, sizes) and
decides which documents to read using the read_full_document tool.  It runs
an iterative Reason + Act loop (via AgentLoop) until it produces a final
answer or hits the iteration cap.

Provider is selected via the LLM_PROVIDER env var (see config.py).
"""

from __future__ import annotations

from dotenv import load_dotenv

from taxonomy_rag.llm.loop import AgentLoop
from taxonomy_rag.llm.provider import get_completion_kwargs
from taxonomy_rag.readers.base import AttachmentInfo
from taxonomy_rag.readers.registry import default_registry
from taxonomy_rag.tools.attachment.read_full import ReadFullDocument
from taxonomy_rag.tools.base import ToolKit
from taxonomy_rag.tracing.base import NullTracer


def get_agent() -> "ReActAgent":
    load_dotenv()
    return ReActAgent()


class ReActAgent:
    """Tool-using agent that selectively reads attachments to answer questions.

    Loop (delegated to AgentLoop):
      1. Send question + attachment list (metadata only) to the model.
      2. If the model calls a tool → execute it, return result, continue.
      3. If the model produces a text response → return it as the answer.
      4. Cap at MAX_ITERATIONS to prevent infinite loops.

    The ToolKit is built fresh per answer() call because ReadFullDocument
    holds a per-question attachment path map.
    """

    def __init__(self) -> None:
        self._loop = AgentLoop(completion_kwargs=get_completion_kwargs())
        self.registry = default_registry()

    def answer(
        self,
        question: str,
        context: str = "",
        prompt: str = "",
        attachments: list[AttachmentInfo] = [],
        tracer: NullTracer = NullTracer(),
    ) -> str:
        path_map = {a.name: a.path for a in attachments}
        toolkit = ToolKit([ReadFullDocument(path_map, self.registry)])

        parts: list[str] = []
        if context:
            parts.append(f"Context:\n{context}")
        if attachments:
            att_lines = [
                f"  - {a.name} ({a.file_type}, {a.size_bytes // 1024} KB)"
                for a in attachments
            ]
            parts.append("Available attachments:\n" + "\n".join(att_lines))
        parts.append(f"Question:\n{question}")
        user_content = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": prompt or "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]

        return self._loop.run(messages=messages, toolkit=toolkit, tracer=tracer)
