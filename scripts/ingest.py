"""CLI entry point for document ingestion.

Usage:
    uv run python scripts/ingest.py <file_path>

The parser and chunker are selected based on settings.default_chunker and
the file extension. Implement the parsers and chunkers in
src/taxonomy_rag/ingestion/ before running this script.
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from taxonomy_rag.config import settings  # noqa: E402
from taxonomy_rag.db.connection import get_pool  # noqa: E402
from taxonomy_rag.db.repository import DocumentRepository  # noqa: E402
from taxonomy_rag.embeddings.embedder import Embedder  # noqa: E402
from taxonomy_rag.ingestion.pipeline import IngestionPipeline  # noqa: E402


def _get_parser(file_path: str):
    """Return the appropriate parser for the given file."""
    if file_path.lower().endswith(".pdf"):
        from taxonomy_rag.ingestion.parsers.pdf import PDFParser
        return PDFParser()
    if file_path.lower().endswith(".xlsx"):
        from taxonomy_rag.ingestion.parsers.spreadsheet import SpreadsheetParser
        return SpreadsheetParser()
    raise ValueError(f"No parser available for: {file_path}")


def _get_chunker(strategy: str):
    """Return the chunker for the given strategy name."""
    if strategy == "naive":
        from taxonomy_rag.ingestion.chunkers.naive import NaiveChunker
        return NaiveChunker(
            chunk_size=settings.default_chunk_size,
            chunk_overlap=settings.default_chunk_overlap,
        )
    if strategy == "structural":
        from taxonomy_rag.ingestion.chunkers.structural import StructuralChunker
        return StructuralChunker()
    if strategy == "hierarchical":
        from taxonomy_rag.ingestion.chunkers.hierarchical import HierarchicalChunker
        return HierarchicalChunker()
    raise ValueError(f"Unknown chunker strategy: {strategy!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a document into the taxonomy RAG database.")
    parser.add_argument("file", help="Path to the PDF or XLSX file to ingest")
    parser.add_argument(
        "--chunker",
        default=settings.default_chunker,
        choices=["naive", "structural", "hierarchical"],
        help="Chunking strategy to use (default: %(default)s)",
    )
    args = parser.parse_args()

    doc_parser = _get_parser(args.file)
    chunker = _get_chunker(args.chunker)
    embedder = Embedder()
    repo = DocumentRepository()

    pipeline = IngestionPipeline(
        parser=doc_parser,
        chunker=chunker,
        embedder=embedder,
        repo=repo,
    )

    print(f"Ingesting: {args.file}")
    print(f"Chunker:   {args.chunker}")

    try:
        result = pipeline.run(args.file)
    except NotImplementedError as e:
        print(f"\nNot implemented yet: {e}", file=sys.stderr)
        print("Implement the parser/chunker in src/taxonomy_rag/ingestion/ first.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDocument ID:    {result.document_id}")
    print(f"Chunks stored:  {result.chunks_stored}")
    if result.errors:
        print(f"Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
