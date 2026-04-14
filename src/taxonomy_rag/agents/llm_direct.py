"""Bare LLM agent — direct Anthropic API call, no retrieval.

This is the bare-LLM baseline: the system prompt is the only guidance,
and the question (plus any inline context) is the only input. No documents
are retrieved. Results represent what the model knows from training alone.

Environment variables read (via python-dotenv / .env):
    ANTHROPIC_API_KEY   — required
    ANTHROPIC_MODEL     — defaults to claude-haiku-4-5-20251001
"""

from __future__ import annotations

import os

import anthropic

from taxonomy_rag.readers.base import AttachmentInfo


_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 1024


def get_agent() -> "LLMDirectAgent":
    return LLMDirectAgent()


class LLMDirectAgent:
    """Calls the Anthropic Messages API with no retrieval or tool use.

    The system prompt comes from the prompt file passed to the eval script.
    The user message is the question, prepended by context if present.
    Attachments are acknowledged but not read — this agent has no tools.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL)

    def answer(
        self,
        question: str,
        context: str = "",
        prompt: str = "",
        attachments: list[AttachmentInfo] = [],
    ) -> str:
        parts = []
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"Question:\n{question}")
        user_content = "\n\n".join(parts)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": user_content}],
        )
        return message.content[0].text
