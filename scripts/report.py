"""Regenerate report.html for an existing eval run directory.

Usage:
    uv run python scripts/report.py runs/20260419_172232_react_naive_corpus_hybrid_rag_simple_v1/

Reads metadata.json, outcomes.csv, and traces/*.json from the run directory
and writes a fresh report.html. Useful for re-styling old runs or runs that
were produced before report generation was added to evaluate.py.

No agent or DB imports — only stdlib + taxonomy_rag.eval_report.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/report.py <run-directory>", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    if not run_dir.is_dir():
        print(f"Error: not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # ── metadata.json ──────────────────────────────────────────────────────
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        print(f"Error: metadata.json not found in {run_dir}", file=sys.stderr)
        sys.exit(1)
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    run_id = metadata["run_id"]

    # ── outcomes.csv ───────────────────────────────────────────────────────
    csv_path = run_dir / "outcomes.csv"
    if not csv_path.exists():
        print(f"Error: outcomes.csv not found in {run_dir}", file=sys.stderr)
        sys.exit(1)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    # ── traces/*.json ──────────────────────────────────────────────────────
    traces: dict[str, dict] = {}
    traces_dir = run_dir / "traces"
    if traces_dir.exists():
        for trace_file in sorted(traces_dir.glob("*_trace.json")):
            with open(trace_file, encoding="utf-8") as f:
                t = json.load(f)
            traces[t["question_id"]] = t

    # ── generate report ────────────────────────────────────────────────────
    from dotenv import load_dotenv
    load_dotenv()
    from taxonomy_rag.eval_report import build_report_html

    html = build_report_html(run_id, metadata, rows, traces)
    report_path = run_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report written: {report_path.relative_to(REPO_ROOT)}")
    print(f"  {len(rows)} questions, {len(traces)} traces")


if __name__ == "__main__":
    main()
