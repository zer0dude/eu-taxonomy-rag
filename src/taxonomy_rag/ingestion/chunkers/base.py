from typing import Protocol

from taxonomy_rag.ingestion.models import Chunk, ParsedDocument


class Chunker(Protocol):
    """Interface that all chunking strategies must implement.

    Add new strategies as new files in this directory — never modify existing ones.
    """

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Split a ParsedDocument into a list of Chunks ready for embedding."""
        ...
