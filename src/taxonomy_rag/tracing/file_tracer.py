"""FileTracer — accumulates a structured trace in memory and writes JSON on save().

One FileTracer instance is created per question by evaluate.py. The agent
calls log_* methods during its reasoning loop; evaluate.py calls save() after
answer() returns.

The amount of data stored per tool call is controlled by LogLevel:
  full      — complete result text
  truncated — first N chars + total char count
  metadata  — char count only, no text
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from taxonomy_rag.tracing.base import LogLevel


class FileTracer:
    """Accumulates trace data and writes it as JSON to a file."""

    def __init__(
        self,
        question_id: str,
        log_level: LogLevel,
        truncate_chars: int = 500,
    ) -> None:
        self._question_id = question_id
        self._log_level = log_level
        self._truncate_chars = truncate_chars
        self._data: dict[str, Any] = {
            "question_id": question_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "log_level": log_level.value,
            "input": {},
            "iterations": [],
            "final_answer": None,
            "total_iterations": 0,
            "duration_seconds": None,
        }
        self._current_iteration: dict | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record_input(
        self,
        question: str,
        context: str,
        attachments: list[Any],
    ) -> None:
        self._data["input"] = {
            "question": question,
            "context": context,
            "attachments": [
                {
                    "name": a.name,
                    "file_type": a.file_type,
                    "size_bytes": a.size_bytes,
                }
                for a in attachments
            ],
        }

    def log_reasoning(self, iteration_index: int, text: str) -> None:
        """Start a new iteration record with the LLM's reasoning text."""
        self._current_iteration = {
            "index": iteration_index,
            "reasoning": text,
            "tool_calls": [],
        }
        self._data["iterations"].append(self._current_iteration)

    def log_tool_call(
        self,
        iteration_index: int,
        tool_name: str,
        tool_input: dict,
        result: str,
        duration_ms: float,
    ) -> None:
        """Append a tool call to the current iteration, respecting log level."""
        # Ensure the current iteration exists (may be called without prior reasoning log)
        if self._current_iteration is None or self._current_iteration["index"] != iteration_index:
            self._current_iteration = {
                "index": iteration_index,
                "reasoning": "",
                "tool_calls": [],
            }
            self._data["iterations"].append(self._current_iteration)

        result_chars = len(result)
        entry: dict[str, Any] = {
            "tool": tool_name,
            "input": tool_input,
            "result_chars": result_chars,
            "duration_ms": round(duration_ms, 1),
        }

        if self._log_level == LogLevel.FULL:
            entry["result"] = result
            entry["result_truncated"] = False
        elif self._log_level == LogLevel.TRUNCATED:
            truncated = result_chars > self._truncate_chars
            entry["result_preview"] = result[: self._truncate_chars]
            entry["result_truncated"] = truncated
        # METADATA: no text fields added — result_chars is sufficient

        self._current_iteration["tool_calls"].append(entry)

    def record_output(self, final_answer: str, duration_seconds: float) -> None:
        self._data["final_answer"] = final_answer
        self._data["total_iterations"] = len(self._data["iterations"])
        self._data["duration_seconds"] = round(duration_seconds, 3)

    def save(self, path: str | Path) -> None:
        """Write accumulated trace to disk as indented JSON (utf-8).

        No-op if record_input was never called (guards against partial traces).
        """
        if not self._data["input"]:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
