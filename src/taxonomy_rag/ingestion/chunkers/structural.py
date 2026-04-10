"""Structural chunker — splits at article and annex boundaries.

STUB — implement this file to enable structure-aware chunking.

Suggested approach:
    Use regex to detect headings like "Article 1", "ANNEX I", "Section 3.2" in
    the concatenated page text. Split the text at each boundary, keeping the
    heading as the first line of each chunk.
    Each Chunk.metadata should include: article, annex, section where detectable.
"""

from taxonomy_rag.ingestion.models import Chunk, ParsedDocument


class StructuralChunker:
    """Chunks a document by splitting at article and annex section boundaries.

    Produces variable-size chunks aligned with the document's logical structure,
    which tends to give cleaner retrieval for regulatory texts.
    """

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        raise NotImplementedError(
            "StructuralChunker.chunk: implement boundary detection and splitting here. "
            "Target boundaries: 'Article N', 'ANNEX N', 'Section N.M'. "
            "See module docstring for a starting point."
        )
