"""CLI entry point for ingesting a single document.

Usage:
    uv run python scripts/ingest.py <file_path> [--strategy naive_pdf]
    uv run python scripts/ingest.py --list-strategies

The strategy name is stored in every chunk's metadata as ``ingestion_strategy``
together with a unique ``ingest_run_id``, so results can be filtered and
compared across experiments.
"""

import argparse
import sys
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from taxonomy_rag.ingestion.strategies.registry import DEFAULT_REGISTRY  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a single document into the taxonomy RAG database."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to the file to ingest (PDF, XLSX, …)",
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_REGISTRY.names()[0],
        choices=DEFAULT_REGISTRY.names(),
        help="Ingestion strategy to use (default: %(default)s)",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Print all available strategy names and exit",
    )
    args = parser.parse_args()

    if args.list_strategies:
        for name in DEFAULT_REGISTRY.names():
            strategy = DEFAULT_REGISTRY.get(name)
            print(f"  {name}  —  {strategy.description}")
        return

    if not args.file:
        parser.error("file argument is required unless --list-strategies is used")

    strategy = DEFAULT_REGISTRY.get(args.strategy)

    if not strategy.supports(args.file):
        print(
            f"ERROR: strategy {args.strategy!r} does not support this file type: {args.file}",
            file=sys.stderr,
        )
        sys.exit(1)

    ingest_run_id = str(uuid4())
    pipeline = strategy.build_pipeline()

    print(f"Ingesting:  {args.file}")
    print(f"Strategy:   {args.strategy}")
    print(f"Run ID:     {ingest_run_id}")

    try:
        result = pipeline.run(args.file, ingest_run_id=ingest_run_id)
    except NotImplementedError as e:
        print(f"\nNot implemented yet: {e}", file=sys.stderr)
        print(
            "Implement the parser/chunker in src/taxonomy_rag/ingestion/ first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDocument ID:    {result.document_id}")
    print(f"Chunks stored:  {result.chunks_stored}")
    if result.errors:
        print(f"Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
