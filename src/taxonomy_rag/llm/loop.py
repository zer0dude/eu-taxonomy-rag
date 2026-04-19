"""AgentLoop — shared LiteLLM tool-use loop for all ReAct-style agents.

Handles the iterative reason-act cycle:
  1. Send messages + tools to litellm.completion()
  2. If finish_reason == "tool_calls": execute tools, append results, loop
  3. If finish_reason == "stop": return the text content
  4. Cap at max_iterations to prevent runaway loops

This class is intentionally stateless between run() calls — all per-call
state lives in local variables inside run(). Instantiate once per agent,
reuse across many run() calls.

Integrates with the existing Tracer protocol (NullTracer by default).
"""

from __future__ import annotations

import json
import time
from typing import Any

import litellm

from taxonomy_rag.tools.base import ToolKit
from taxonomy_rag.tracing.base import NullTracer, Tracer

_MAX_TOKENS = 2048
_MAX_ITERATIONS = 10


class AgentLoop:
    """Provider-agnostic tool-use loop using litellm.completion().

    Args:
        completion_kwargs: Base kwargs for litellm.completion() — must include
                           ``"model"``.  Typically from ``get_completion_kwargs()``.
        max_tokens:        Token limit per model call.
        max_iterations:    Safety cap on the number of reasoning iterations.
    """

    def __init__(
        self,
        completion_kwargs: dict[str, Any],
        max_tokens: int = _MAX_TOKENS,
        max_iterations: int = _MAX_ITERATIONS,
    ) -> None:
        self._kwargs = completion_kwargs
        self._max_tokens = max_tokens
        self._max_iterations = max_iterations

    def run(
        self,
        messages: list[dict],
        toolkit: ToolKit,
        tracer: Tracer = NullTracer(),
    ) -> str:
        """Execute the tool-use loop and return the final text answer.

        Args:
            messages:  Pre-built message list in OpenAI format (system + user).
                       AgentLoop appends assistant and tool messages as the loop runs.
                       A local copy is made so the caller's list is not mutated.
            toolkit:   ToolKit instance for this call (may hold per-call state).
            tracer:    Tracing sink; NullTracer discards all events.

        Returns:
            The final text response from the model, or a sentinel string if the
            loop exits due to max_iterations or an unexpected finish_reason.
        """
        msgs = list(messages)  # local copy — do not mutate caller's list

        for iteration in range(self._max_iterations):
            response = litellm.completion(
                **self._kwargs,
                messages=msgs,
                tools=toolkit.to_litellm_schema(),
                max_tokens=self._max_tokens,
            )

            choice = response.choices[0]
            message = choice.message

            # content is None when finish_reason == "tool_calls" — always guard
            reasoning = message.content or ""
            tracer.log_reasoning(iteration + 1, reasoning)

            if choice.finish_reason == "stop":
                return reasoning

            if choice.finish_reason == "tool_calls":
                # Append the assistant turn; LiteLLM returns a Pydantic model
                try:
                    msgs.append(message.model_dump())
                except AttributeError:
                    msgs.append(dict(message))

                for call in message.tool_calls:
                    tool_name = call.function.name
                    # arguments is a JSON-encoded string in OpenAI format
                    tool_input = json.loads(call.function.arguments)

                    t0 = time.monotonic()
                    result = toolkit.run(tool_name, tool_input)
                    duration_ms = (time.monotonic() - t0) * 1000

                    tracer.log_tool_call(
                        iteration + 1,
                        tool_name,
                        tool_input,
                        result,
                        duration_ms,
                    )

                    msgs.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    })
                continue

            # Unexpected finish_reason (e.g. "length", "content_filter")
            return (
                message.content
                or f"[AgentLoop] Unexpected finish_reason: {choice.finish_reason}"
            )

        return "[AgentLoop] Max iterations reached without a final answer."
