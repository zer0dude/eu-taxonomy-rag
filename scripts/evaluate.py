"""CLI entry point for evaluation against the golden dataset.

Usage:
    uv run python scripts/evaluate.py [--rag naive|hybrid|advanced]

Runs the EvaluationHarness against eval/golden_dataset.json and prints a
comparison table. Exits gracefully with a message if no documents have been
ingested yet.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from taxonomy_rag.db.repository import DocumentRepository  # noqa: E402
from taxonomy_rag.eval.harness import EvaluationHarness  # noqa: E402


DATASET_PATH = Path(__file__).parent.parent / "eval" / "golden_dataset.json"


def _get_rag_pipeline(strategy: str):
    if strategy == "naive":
        from taxonomy_rag.rag.naive import NaiveRAG
        return NaiveRAG()
    if strategy == "hybrid":
        from taxonomy_rag.rag.hybrid import HybridRAG
        return HybridRAG()
    if strategy == "advanced":
        from taxonomy_rag.rag.advanced import AdvancedRAG
        return AdvancedRAG()
    raise ValueError(f"Unknown RAG strategy: {strategy!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline against golden dataset.")
    parser.add_argument(
        "--rag",
        default="naive",
        choices=["naive", "hybrid", "advanced"],
        help="RAG pipeline to evaluate (default: %(default)s)",
    )
    args = parser.parse_args()

    # Check whether any documents have been ingested
    repo = DocumentRepository()
    try:
        docs = repo.get_all(limit=1)
    except Exception as e:
        print(f"Could not connect to the database: {e}", file=sys.stderr)
        print("Run `docker compose up -d` first.", file=sys.stderr)
        sys.exit(1)

    if not docs:
        print("No documents ingested yet.")
        print("Run `uv run python scripts/ingest.py <file>` first.")
        sys.exit(0)

    rag = _get_rag_pipeline(args.rag)
    harness = EvaluationHarness(dataset_path=str(DATASET_PATH), rag_pipeline=rag)
    results = harness.run()
    harness.report(results)


if __name__ == "__main__":
    main()
