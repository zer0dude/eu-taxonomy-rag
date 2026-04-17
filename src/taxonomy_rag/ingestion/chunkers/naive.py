"""Naive fixed-size word-window chunker.

Splits a document by flattening all pages into a sequence of (word, page_num)
pairs, then sliding a window of `chunk_size` words with `chunk_overlap` word
overlap. No sentence or structural awareness — pure sliding window.

This is the baseline to compare future chunking strategies against.
Tokenisation is whitespace-split (no tiktoken dependency required).
"""

from __future__ import annotations

from taxonomy_rag.ingestion.models import Chunk, ParsedDocument


class NaiveChunker:
    """Splits a document into fixed-size word-window chunks with overlap.

    Args:
        chunk_size:    Number of words per chunk.
        chunk_overlap: Number of words repeated at the start of each
                       successive chunk (must be < chunk_size).
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        # Build a flat list of (word, 1-based page number) pairs.
        word_page_pairs: list[tuple[str, int]] = []
        for page_num, page_text in enumerate(document.pages, start=1):
            words = page_text.split()
            word_page_pairs.extend((w, page_num) for w in words)

        if not word_page_pairs:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        start = 0

        while start < len(word_page_pairs):
            window = word_page_pairs[start : start + self.chunk_size]

            content = " ".join(w for w, _ in window)
            pages_in_window = [p for _, p in window]
            first_page = pages_in_window[0]
            last_page = pages_in_window[-1]
            page_range = (
                f"p{first_page}" if first_page == last_page
                else f"p{first_page}-p{last_page}"
            )

            chunks.append(
                Chunk(
                    content=content,
                    metadata={
                        "source":          document.source_path,
                        "document_id":     document.document_id,
                        "document_type":   document.document_type,
                        "chunk_strategy":  "naive",
                        "chunk_index":     len(chunks),
                        "page_range":      page_range,
                    },
                )
            )

            start += step

        return chunks
