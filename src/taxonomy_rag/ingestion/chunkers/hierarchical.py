"""Hierarchical chunker — small leaf chunks with parent article context.

STUB — implement this file to enable hierarchical chunking.

Suggested approach:
    1. First apply structural splitting to get article-level parent chunks.
    2. Then split each parent into smaller leaf chunks (e.g. paragraph level).
    3. Each leaf Chunk.metadata['parent_content'] stores the parent article text,
       so the retriever can optionally return parent context along with the leaf.
    This is also known as the "small-to-big" or "parent document retriever" pattern.
"""

from taxonomy_rag.ingestion.models import Chunk, ParsedDocument


class HierarchicalChunker:
    """Produces small leaf chunks that carry a reference to their parent article.

    Retrieval happens on the small chunks (high precision), but the LLM receives
    the full parent article as context (high recall). Requires storing both leaf
    and parent chunks, or embedding a parent_content field in metadata.
    """

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        raise NotImplementedError(
            "HierarchicalChunker.chunk: implement two-level splitting here. "
            "Step 1: structural split → parent articles. "
            "Step 2: split each parent into leaf chunks, attaching parent text to metadata. "
            "See module docstring for a starting point."
        )
