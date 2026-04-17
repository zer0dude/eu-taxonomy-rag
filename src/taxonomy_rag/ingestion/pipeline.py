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

    Args:
        strategy_name: The IngestionStrategy.name that built this pipeline.
            Injected into every chunk's metadata as ``ingestion_strategy``.
            If None, the field is omitted (backwards-compatible).
    """

    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        embedder: Embedder,
        repo: DocumentRepository,
        strategy_name: str | None = None,
    ) -> None:
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.repo = repo
        self.strategy_name = strategy_name

    def run(self, file_path: str, ingest_run_id: str | None = None) -> IngestionResult:
        """Ingest a single document file.

        Steps:
            1. parser.parse(file_path) → ParsedDocument
            2. chunker.chunk(document)  → list[Chunk]
            3. Inject ingestion_strategy / ingest_run_id into chunk metadata
            4. embedder.embed_batch([c.content for c in chunks]) → list[list[float]]
            5. repo.insert(...) for each chunk

        Args:
            ingest_run_id: Optional UUID for the enclosing corpus run.
                Stored in metadata so an entire run can be queried or deleted.

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

        # Step 3: inject tracking metadata
        for chunk in chunks:
            if self.strategy_name:
                chunk.metadata["ingestion_strategy"] = self.strategy_name
            if ingest_run_id:
                chunk.metadata["ingest_run_id"] = ingest_run_id

        # Step 4: embed all chunks in one batch
        embeddings = self.embedder.embed_batch([c.content for c in chunks])

        # Step 5: store
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
