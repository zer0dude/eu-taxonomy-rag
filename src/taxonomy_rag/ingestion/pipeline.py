from __future__ import annotations

from dataclasses import dataclass, field

from taxonomy_rag.db.repository import DocumentRepository
from taxonomy_rag.embeddings.embedder import Embedder
from taxonomy_rag.ingestion.chunkers.base import Chunker
from taxonomy_rag.ingestion.parsers.base import DocumentParser


@dataclass
class IngestionResult:
    document_id: str
    chunks_stored: int
    errors: list[str] = field(default_factory=list)


class IngestionPipeline:
    """Orchestrates the full ingestion flow: parse → chunk → embed → store.

    Each component is injected so that different strategies can be swapped in
    without modifying this class.
    """

    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        embedder: Embedder,
        repo: DocumentRepository,
    ) -> None:
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.repo = repo

    def run(self, file_path: str) -> IngestionResult:
        """Ingest a single document file.

        Steps:
            1. parser.parse(file_path) → ParsedDocument
            2. chunker.chunk(document)  → list[Chunk]
            3. embedder.embed_batch([c.content for c in chunks]) → list[list[float]]
            4. repo.insert(...) for each chunk

        Returns an IngestionResult with counts and any per-chunk errors.
        """
        # Step 1: parse
        document = self.parser.parse(file_path)

        # Step 2: chunk
        chunks = self.chunker.chunk(document)

        if not chunks:
            return IngestionResult(
                document_id=document.document_id,
                chunks_stored=0,
                errors=["Chunker returned no chunks — check parser output."],
            )

        # Step 3: embed all chunks in one batch
        embeddings = self.embedder.embed_batch([c.content for c in chunks])

        # Step 4: store
        errors: list[str] = []
        stored = 0
        for chunk, embedding in zip(chunks, embeddings):
            try:
                self.repo.insert(
                    content=chunk.content,
                    embedding=embedding,
                    metadata=chunk.metadata,
                )
                stored += 1
            except Exception as exc:
                errors.append(f"Chunk insert failed: {exc}")

        return IngestionResult(
            document_id=document.document_id,
            chunks_stored=stored,
            errors=errors,
        )
