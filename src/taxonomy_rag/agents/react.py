"""ReAct agent — Anthropic tool-use loop with attachment reading.

The agent receives a list of attachment metadata (names, types, sizes) and
decides which documents to read using the read_full_document tool. It runs
an iterative Reason + Act loop until it produces a final answer or hits the
iteration cap.

Environment variables read (via python-dotenv / .env):
    ANTHROPIC_API_KEY   — required
    ANTHROPIC_MODEL     — defaults to claude-haiku-4-5-20251001
"""

from __future__ import annotations

import os
import time

import anthropic

from taxonomy_rag.readers.base import AttachmentInfo
from taxonomy_rag.readers.registry import default_registry
from taxonomy_rag.tools.attachment.read_full import ReadFullDocument
from taxonomy_rag.tools.base import ToolKit
from taxonomy_rag.tracing.base import NullTracer


_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 2048
_MAX_ITERATIONS = 10


def get_agent() -> "ReActAgent":
    from dotenv import load_dotenv
    load_dotenv()
    return ReActAgent()


class ReActAgent:
    """Tool-using agent that selectively reads attachments to answer questions.

    Loop:
      1. Send question + attachment list (metadata only) to the model.
      2. If the model calls a tool → execute it, return result, continue.
      3. If the model produces a text response → return it as the answer.
      4. Cap at MAX_ITERATIONS to prevent infinite loops.

    The ToolKit is built fresh per answer() call because it holds a
    per-question attachment path map.
    """

    MAX_ITERATIONS = _MAX_ITERATIONS

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL)
        self.registry = default_registry()

    def answer(
        self,
        question: str,
        context: str = "",
        prompt: str = "",
        attachments: list[AttachmentInfo] = [],
        tracer: NullTracer = NullTracer(),
    ) -> str:
        # Build per-question toolkit with the resolved path map
        path_map = {a.name: a.path for a in attachments}
        toolkit = ToolKit([ReadFullDocument(path_map, self.registry)])

        # Build the initial user message
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

        messages: list[dict] = [{"role": "user", "content": user_content}]

        for iteration in range(self.MAX_ITERATIONS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=prompt or "You are a helpful assistant.",
                tools=toolkit.to_anthropic_schema(),
                messages=messages,
            )

            # Extract reasoning text from text blocks in this response
            reasoning = " ".join(
                b.text for b in response.content if b.type == "text"
            )
            tracer.log_reasoning(iteration + 1, reasoning)

            if response.stop_reason == "end_turn":
                final = next(
                    (b.text for b in response.content if hasattr(b, "text")),
                    "",
                )
                return final

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        t0 = time.monotonic()
                        result = toolkit.run(block.name, block.input)
                        duration_ms = (time.monotonic() - t0) * 1000
                        tracer.log_tool_call(
                            iteration + 1,
                            block.name,
                            block.input,
                            result,
                            duration_ms,
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason — return whatever text is available
            return next(
                (b.text for b in response.content if hasattr(b, "text")),
                f"[ReAct] Unexpected stop_reason: {response.stop_reason}",
            )

        return "[ReAct] Max iterations reached without a final answer."
