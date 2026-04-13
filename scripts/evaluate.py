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


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _get_agent(name: str):
    if name == "mock":
        from taxonomy_rag.agents.mock import MockAgent
        return MockAgent()
    raise ValueError(f"Unknown agent: {name!r}. Available: mock")


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
        required=True,
        help="Path to prompt text file (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=["mock"],
        help="Agent to use",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = REPO_ROOT / questions_path

    prompt_path = Path(args.prompt)
    if not prompt_path.is_absolute():
        prompt_path = REPO_ROOT / prompt_path

    if not questions_path.exists():
        print(f"Error: questions file not found: {questions_path}", file=sys.stderr)
        sys.exit(1)
    if not prompt_path.exists():
        print(f"Error: prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    with open(prompt_path, encoding="utf-8") as f:
        prompt = f.read()

    agent = _get_agent(args.agent)
    question_set = _question_set_name(questions_path)

    # Build run folder name
    now = datetime.now(tz=timezone.utc)
    run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{args.agent}_{question_set}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run the agent over each question
    t_start = time.monotonic()
    rows: list[dict] = []
    for item in questions:
        context = item.get("context", "")
        question = item["question"]
        answer = agent.answer(question=question, context=context, prompt=prompt)

        notes = _get_evaluator_notes(item)
        rows.append({
            "question_id": item["id"],
            "difficulty": item.get("difficulty", ""),
            "tags": "; ".join(item.get("tags", [])),
            "adversarial_type": item.get("adversarial_type", ""),
            "context": context,
            "question": question,
            "what_to_look_for": " | ".join(notes["what_to_look_for"]),
            "key_citations": " | ".join(notes["key_citations"]),
            "reference_answer": notes["reference_answer"],
            "agent_answer": answer,
            "human_score": "",
            "human_notes": "",
        })
    duration = time.monotonic() - t_start

    # Write metadata.json
    metadata = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "agent": args.agent,
        "questions_file": str(questions_path.relative_to(REPO_ROOT)),
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
        "total_questions": len(rows),
        "duration_seconds": round(duration, 3),
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
        "what_to_look_for",
        "key_citations",
        "reference_answer",
        "agent_answer",
        "human_score",
        "human_notes",
    ]
    csv_path = run_dir / "outcomes.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary to stdout
    print(f"Run:       {run_id}")
    print(f"Agent:     {args.agent}")
    print(f"Questions: {questions_path.relative_to(REPO_ROOT)}  ({len(rows)} items)")
    print(f"Prompt:    {prompt_path.relative_to(REPO_ROOT)}")
    print(f"Duration:  {duration:.3f}s")
    print(f"Output:    {run_dir.relative_to(REPO_ROOT)}/")
    print(f"           +-- metadata.json")
    print(f"           +-- outcomes.csv")


if __name__ == "__main__":
    main()
