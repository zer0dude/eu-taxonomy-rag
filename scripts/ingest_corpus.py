"""Bulk ingestion script for the full eu-tax-docs corpus.

Walks a directory tree, ingests every file the chosen strategy supports,
and reports a summary. Each run generates a unique ``ingest_run_id`` stored
in chunk metadata so the run can be queried or deleted cleanly.

Usage:
    uv run python scripts/ingest_corpus.py [options]
    uv run python scripts/ingest_corpus.py --list-strategies

Options:
    --strategy   Ingestion strategy name (default: first registered strategy)
    --dir        Root directory to scan (default: data/raw/eu-tax-docs)
    --dry-run    List files that would be ingested without writing to the DB

Re-ingestion: the documents table has no unique constraint on document_id, so
running twice inserts duplicate chunks. Use the run ID printed at start to
delete the old run first:
    DELETE FROM documents WHERE metadata->>'ingest_run_id' = '<old-run-id>';
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from taxonomy_rag.db.connection import get_pool  # noqa: E402
from taxonomy_rag.ingestion.strategies.registry import DEFAULT_REGISTRY  # noqa: E402


_DEFAULT_DIR = Path(__file__).parent.parent / "data" / "raw" / "eu-tax-docs"


def _collect_files(root: Path) -> list[Path]:
    """Return all files under root recursively, sorted for reproducibility."""
    return sorted(p for p in root.rglob("*") if p.is_file())


def _check_existing_rows() -> int:
    with get_pool().connection() as conn:
        row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
    return row[0] if row else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the eu-tax-docs corpus into the taxonomy RAG database."
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_REGISTRY.names()[0],
        choices=DEFAULT_REGISTRY.names(),
        help="Ingestion strategy (default: %(default)s)",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=_DEFAULT_DIR,
        help="Root directory to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be ingested without writing to the DB",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Print all available strategy names and exit",
    )
    args = parser.parse_args()

    if args.list_strategies:
        for name in DEFAULT_REGISTRY.names():
            s = DEFAULT_REGISTRY.get(name)
            print(f"  {name}  —  {s.description}")
        return

    root: Path = args.dir.resolve()
    if not root.exists():
        print(f"ERROR: directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    strategy = DEFAULT_REGISTRY.get(args.strategy)
    all_files = _collect_files(root)
    supported = [f for f in all_files if strategy.supports(str(f))]
    skipped = len(all_files) - len(supported)

    print(f"Strategy:   {args.strategy}")
    print(f"Directory:  {root}")
    print(f"Files found:    {len(all_files)}  ({skipped} skipped — not supported by strategy)")
    print(f"Files to ingest: {len(supported)}")
    print()

    if not supported:
        print("Nothing to ingest.")
        return

    if args.dry_run:
        for f in supported:
            print(f"  {f.relative_to(root)}")
        print(f"\nDry run complete — {len(supported)} file(s) would be ingested.")
        return

    # Warn if the table already has data.
    existing = _check_existing_rows()
    if existing > 0:
        print(
            f"WARNING: documents table already contains {existing:,} row(s).\n"
            "         Running again will insert duplicate chunks.\n"
            "         Delete the old run first if needed (see notes.md — DB management).\n"
        )

    ingest_run_id = str(uuid4())
    print(f"Run ID:     {ingest_run_id}\n")

    total_chunks = 0
    total_errors = 0
    failed_files: list[str] = []

    for f in supported:
        rel = str(f.relative_to(root))
        pipeline = strategy.build_pipeline()
        try:
            result = pipeline.run(str(f), ingest_run_id=ingest_run_id)
        except NotImplementedError as exc:
            print(f"  SKIP  {rel}: not implemented — {exc}", file=sys.stderr)
            failed_files.append(rel)
            continue
        except Exception as exc:
            print(f"  ERROR {rel}: {exc}", file=sys.stderr)
            failed_files.append(rel)
            total_errors += 1
            continue

        chunk_errors = len(result.errors)
        total_chunks += result.chunks_stored
        total_errors += chunk_errors

        status = "✓" if not chunk_errors else "~"
        print(f"  {status} {result.document_id} — {result.chunks_stored} chunks", end="")
        if chunk_errors:
            print(f"  ({chunk_errors} chunk errors)", end="")
        print()

    print()
    print("─" * 50)
    print(f"Run ID:           {ingest_run_id}")
    print(f"Files processed:  {len(supported) - len(failed_files)} / {len(supported)}")
    print(f"Total chunks:     {total_chunks:,}")
    if total_errors:
        print(f"Total errors:     {total_errors}")
    if failed_files:
        print(f"Failed files ({len(failed_files)}):")
        for fn in failed_files:
            print(f"  - {fn}")


if __name__ == "__main__":
    main()
