"""Modular evaluation script for EU Taxonomy agents.

Usage:
    uv run python scripts/evaluate.py \\
        --questions eval/simple_v1/questions.json \\
        --prompt prompts/base_v1.txt \\
        --agent mock

Each run is saved to runs/{YYYYMMDD}_{HHMMSS}_{agent}_{question_set}/
containing:
    metadata.json  — run parameters and summary stats
    outcomes.csv   — one row per question with agent answer and empty
                     human_score / human_notes columns for manual review
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
RUNS_DIR = REPO_ROOT / "runs"

# Preferred attachment formats in priority order (first found wins)
_ATTACHMENT_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".csv"]

_LOG_LEVEL_CHOICES = ["full", "truncated", "metadata", "none"]
_LOG_LEVEL_DEFAULT = "full"
_TRUNCATE_CHARS_DEFAULT = 500


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _get_agent(name: str):
    """Dynamically import taxonomy_rag.agents.<name> and call its get_agent().

    Convention: every agent module must expose a module-level get_agent()
    function that returns an object satisfying AgentProtocol. Adding a new
    agent only requires creating a new file — evaluate.py never needs editing.
    """
    import importlib
    from dotenv import load_dotenv
    load_dotenv()
    try:
        module = importlib.import_module(f"taxonomy_rag.agents.{name}")
    except ModuleNotFoundError:
        print(
            f"Error: no agent module 'taxonomy_rag/agents/{name}.py' found.\n"
            f"Create the file and add a get_agent() function to register it.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not hasattr(module, "get_agent"):
        print(
            f"Error: taxonomy_rag/agents/{name}.py has no get_agent() function.\n"
            f"Add 'def get_agent(): return YourAgent()' to the module.",
            file=sys.stderr,
        )
        sys.exit(1)
    return module.get_agent()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _question_set_name(questions_path: Path) -> str:
    """Derive a short display name from the questions file path.

    If the file is named questions.json, use the parent directory name
    (e.g. eval/simple_v1/questions.json → simple_v1).
    Otherwise use the file stem.
    """
    if questions_path.name == "questions.json":
        return questions_path.parent.name
    return questions_path.stem


def _get_evaluator_notes(item: dict) -> dict:
    """Normalise evaluator_notes across question schemas.

    simple_v1 / hard_01 style:
        item["evaluator_notes"] = {what_to_look_for, key_citations, reference_answer}

    golden_dataset (legacy) style:
        item["key_facts"], item["notes"]  (no evaluator_notes key)
    """
    if "evaluator_notes" in item:
        notes = item["evaluator_notes"]
        return {
            "what_to_look_for": notes.get("what_to_look_for", []),
            "key_citations": notes.get("key_citations", []),
            "reference_answer": notes.get("reference_answer", ""),
        }
    # Legacy golden_dataset format
    return {
        "what_to_look_for": item.get("key_facts", []),
        "key_citations": [],
        "reference_answer": "",
    }


def _resolve_attachments(item: dict, questions_path: Path) -> list:
    """Resolve attachment names in a question item to AttachmentInfo objects.

    Convention: attachments live at
        eval/{question_set}/attachments/{question_id}/{name}{ext}

    Prefers .pdf over .docx over .xlsx. Prints a warning if a listed
    attachment cannot be found on disk (non-fatal).

    Returns an empty list if the question has no attachments field.
    """
    from taxonomy_rag.readers.base import AttachmentInfo

    attachment_names: list[str] = item.get("attachments", [])
    if not attachment_names:
        return []

    question_id = item["id"]
    attachment_dir = questions_path.parent / "attachments" / question_id

    result: list[AttachmentInfo] = []
    for name in attachment_names:
        found = False
        for ext in _ATTACHMENT_EXTENSIONS:
            candidate = attachment_dir / f"{name}{ext}"
            if candidate.exists():
                result.append(AttachmentInfo(
                    name=name,
                    file_type=ext.lstrip("."),
                    size_bytes=candidate.stat().st_size,
                    path=str(candidate),
                ))
                found = True
                break
        if not found:
            print(
                f"  Warning: attachment '{name}' listed in question '{question_id}' "
                f"not found under {attachment_dir}",
                file=sys.stderr,
            )

    return result


def _format_attachments_col(attachments: list) -> str:
    """Format attachment list as a readable string for the CSV column."""
    if not attachments:
        return ""
    return "; ".join(
        f"{a.name} ({a.file_type}, {a.size_bytes // 1024} KB)"
        for a in attachments
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an agent over a question set and save results to runs/."
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to questions JSON file (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--prompt",
        required=False,
        default=None,
        help=(
            "Path to prompt text file (relative to repo root or absolute). "
            "If omitted, the agent uses its built-in default prompt."
        ),
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent to use. Must match a module in src/taxonomy_rag/agents/ that exposes get_agent().",
    )
    parser.add_argument(
        "--log-level",
        default=_LOG_LEVEL_DEFAULT,
        choices=_LOG_LEVEL_CHOICES,
        dest="log_level",
        help=(
            "Trace verbosity. full=complete tool results, truncated=first N chars, "
            "metadata=sizes only, none=no trace files. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--truncate-chars",
        type=int,
        default=_TRUNCATE_CHARS_DEFAULT,
        dest="truncate_chars",
        help="Characters to keep per tool result when --log-level=truncated. Default: %(default)s",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = REPO_ROOT / questions_path

    prompt_path: Path | None = None
    prompt: str = ""
    if args.prompt is not None:
        prompt_path = Path(args.prompt)
        if not prompt_path.is_absolute():
            prompt_path = REPO_ROOT / prompt_path
        if not prompt_path.exists():
            print(f"Error: prompt file not found: {prompt_path}", file=sys.stderr)
            sys.exit(1)
        with open(prompt_path, encoding="utf-8") as f:
            prompt = f.read()

    if not questions_path.exists():
        print(f"Error: questions file not found: {questions_path}", file=sys.stderr)
        sys.exit(1)

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    agent = _get_agent(args.agent)
    question_set = _question_set_name(questions_path)

    # Build run folder name
    now = datetime.now(tz=timezone.utc)
    run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{args.agent}_{question_set}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tracing imports — lazy to keep startup fast when log_level=none
    if args.log_level != "none":
        from taxonomy_rag.tracing.base import LogLevel
        from taxonomy_rag.tracing.file_tracer import FileTracer
    from taxonomy_rag.tracing.base import NullTracer

    # Run the agent over each question
    t_start = time.monotonic()
    rows: list[dict] = []
    for item in questions:
        context = item.get("context", "")
        question = item["question"]
        attachments = _resolve_attachments(item, questions_path)

        print(f"[{item['id']}] {question[:80]}")
        if attachments:
            for a in attachments:
                print(f"  attachment: {a.name} ({a.file_type}, {a.size_bytes // 1024} KB)")

        if args.log_level != "none":
            tracer = FileTracer(item["id"], LogLevel(args.log_level), args.truncate_chars)
        else:
            tracer = NullTracer()

        tracer.record_input(question, context, attachments)

        q_start = time.monotonic()
        answer = agent.answer(
            question=question,
            context=context,
            prompt=prompt,
            attachments=attachments,
            tracer=tracer,
        )
        q_duration = time.monotonic() - q_start

        tracer.record_output(answer, q_duration)
        tracer.save(run_dir / "traces" / f"{item['id']}_trace.json")

        usage = tracer.token_totals
        notes = _get_evaluator_notes(item)
        rows.append({
            "question_id": item["id"],
            "difficulty": item.get("difficulty", ""),
            "tags": "; ".join(item.get("tags", [])),
            "adversarial_type": item.get("adversarial_type", ""),
            "context": context,
            "question": question,
            "attachments": _format_attachments_col(attachments),
            "what_to_look_for": " | ".join(notes["what_to_look_for"]),
            "key_citations": " | ".join(notes["key_citations"]),
            "reference_answer": notes["reference_answer"],
            "agent_answer": answer,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "human_score": "",
            "human_notes": "",
        })
    duration = time.monotonic() - t_start

    total_input_tokens = sum(r["input_tokens"] for r in rows)
    total_output_tokens = sum(r["output_tokens"] for r in rows)

    from taxonomy_rag.eval_report import build_report_html

    # Write metadata.json
    metadata = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "agent": args.agent,
        "questions_file": str(questions_path.relative_to(REPO_ROOT)),
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)) if prompt_path else None,
        "log_level": args.log_level,
        "truncate_chars": args.truncate_chars if args.log_level == "truncated" else None,
        "total_questions": len(rows),
        "duration_seconds": round(duration, 3),
        "token_totals": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "note": "Counts may be zero for providers that do not report usage (e.g. Ollama).",
        },
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Write outcomes.csv
    fieldnames = [
        "question_id",
        "difficulty",
        "tags",
        "adversarial_type",
        "context",
        "question",
        "attachments",
        "what_to_look_for",
        "key_citations",
        "reference_answer",
        "agent_answer",
        "input_tokens",
        "output_tokens",
        "human_score",
        "human_notes",
    ]
    csv_path = run_dir / "outcomes.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    # Generate HTML report
    traces: dict[str, dict] = {}
    traces_dir = run_dir / "traces"
    if traces_dir.exists():
        for trace_file in sorted(traces_dir.glob("*_trace.json")):
            with open(trace_file, encoding="utf-8") as f:
                t = json.load(f)
            traces[t["question_id"]] = t

    report_html = build_report_html(run_id, metadata, rows, traces)
    report_path = run_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    # Summary to stdout
    print(f"\nRun:       {run_id}")
    print(f"Agent:     {args.agent}")
    print(f"Questions: {questions_path.relative_to(REPO_ROOT)}  ({len(rows)} items)")
    print(f"Prompt:    {prompt_path.relative_to(REPO_ROOT) if prompt_path else '(agent default)'}")
    print(f"Duration:  {duration:.3f}s")
    print(f"Tokens:    {total_input_tokens} in / {total_output_tokens} out")
    print(f"Output:    {run_dir.relative_to(REPO_ROOT)}/")
    print(f"           +-- metadata.json")
    print(f"           +-- outcomes.csv")
    print(f"           +-- report.html")


if __name__ == "__main__":
    main()
