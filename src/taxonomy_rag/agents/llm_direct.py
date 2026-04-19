"""Bare LLM agent — direct litellm.completion() call, no retrieval.

This is the bare-LLM baseline: the system prompt is the only guidance,
and the question (plus any inline context) is the only input.  No documents
are retrieved.  Results represent what the model knows from training alone.

Provider is selected via the LLM_PROVIDER env var (see config.py).
"""

from __future__ import annotations

import litellm

from taxonomy_rag.llm.provider import get_completion_kwargs
from taxonomy_rag.readers.base import AttachmentInfo
from taxonomy_rag.tracing.base import NullTracer, Tracer

_MAX_TOKENS = 1024


def get_agent() -> "LLMDirectAgent":
    return LLMDirectAgent()


class LLMDirectAgent:
    """Calls litellm.completion() with no retrieval or tool use.

    The system prompt comes from the prompt file passed to the eval script.
    The user message is the question, prepended by context if present.
    Attachments are acknowledged but not read — this agent has no tools.
    """

    def __init__(self) -> None:
        self._completion_kwargs = get_completion_kwargs()

    def answer(
        self,
        question: str,
        context: str = "",
        prompt: str = "",
        attachments: list[AttachmentInfo] | None = None,
        tracer: Tracer = NullTracer(),
    ) -> str:
        parts: list[str] = []
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"Question:\n{question}")
        user_content = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": prompt or "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]

        response = litellm.completion(
            **self._completion_kwargs,
            messages=messages,
            max_tokens=_MAX_TOKENS,
        )
        return response.choices[0].message.content or ""
