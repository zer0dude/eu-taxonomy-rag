"""Naive PDF ingestion strategy.

Composition: PDFParser (pymupdf plain text) + NaiveChunker (sliding word window).

This is the baseline strategy. All future strategies are compared against it.
Add new strategies as new files — never modify this one.
"""

from taxonomy_rag.config import settings
from taxonomy_rag.db.repository import DocumentRepository
from taxonomy_rag.embeddings.embedder import Embedder
from taxonomy_rag.ingestion.chunkers.naive import NaiveChunker
from taxonomy_rag.ingestion.parsers.pdf import PDFParser
from taxonomy_rag.ingestion.pipeline import IngestionPipeline


class NaivePDFStrategy:
    """PDFParser + NaiveChunker (512 words, 50 overlap).

    Extracts plain text page-by-page via pymupdf and splits into fixed-size
    word windows. No structural or semantic awareness.
    """

    name = "naive_pdf"
    description = "PDFParser (pymupdf plain text) + NaiveChunker (512 words, 50 overlap)"

    def supports(self, file_path: str) -> bool:
        return PDFParser().supports(file_path)

    def build_pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            parser=PDFParser(),
            chunker=NaiveChunker(
                chunk_size=settings.default_chunk_size,
                chunk_overlap=settings.default_chunk_overlap,
            ),
            embedder=Embedder(),
            repo=DocumentRepository(),
            strategy_name=self.name,
        )
