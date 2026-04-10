"""Naive fixed-size token chunker.

STUB — implement this file to enable the naive chunking strategy.

Suggested approach:
    Tokenise each page's text, split into windows of `chunk_size` tokens with
    `chunk_overlap` token overlap. Use tiktoken or a simple whitespace splitter.
    Each Chunk.metadata must include: source, document_id, chunk_strategy="naive".
"""

from taxonomy_rag.ingestion.models import Chunk, ParsedDocument


class NaiveChunker:
    """Splits a document into fixed-size token chunks with overlap.

    Args:
        chunk_size:    Maximum number of tokens per chunk.
        chunk_overlap: Number of tokens to repeat at the start of each chunk.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        raise NotImplementedError(
            "NaiveChunker.chunk: implement fixed-size token chunking here. "
            f"Target chunk_size={self.chunk_size}, overlap={self.chunk_overlap}. "
            "See module docstring for a starting point."
        )
