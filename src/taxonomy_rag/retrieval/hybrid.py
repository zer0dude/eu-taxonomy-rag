from __future__ import annotations

from taxonomy_rag.db.repository import DocumentRepository
from taxonomy_rag.embeddings.embedder import Embedder
from taxonomy_rag.retrieval.base import RetrievalResult
from taxonomy_rag.retrieval.scope import CorpusScope
from taxonomy_rag.tracing.base import NullTracer, Tracer


class HybridRetrieval:
    """Retrieval: embed query → RRF(BM25 full-text + cosine vector) → RetrievalResult list.

    Combines two ranked lists via Reciprocal Rank Fusion entirely inside a single
    SQL CTE (DocumentRepository.hybrid_search). Better than cosine-only for queries
    with strong keyword signal (specific regulation codes, article numbers, etc.).

    Compose with a CorpusScope and wrap in SearchCorpusTool to expose as an agent tool.
    """

    def __init__(
        self,
        repo: DocumentRepository | None = None,
        embedder: Embedder | None = None,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self.repo = repo or DocumentRepository()
        self.embedder = embedder or Embedder()
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        scope: CorpusScope | None = None,
        tracer: Tracer = NullTracer(),
    ) -> list[RetrievalResult]:
        embedding = self.embedder.embed(query)
        metadata_filter = scope.to_metadata_filter() if scope else None
        docs = self.repo.hybrid_search(
            query,
            embedding,
            self.top_k,
            self.rrf_k,
            metadata_filter=metadata_filter,
        )
        return [
            RetrievalResult(
                doc_id=d["id"],
                content=d["content"],
                score=float(d["score"]),
                metadata=d.get("metadata") or {},
            )
            for d in docs
        ]
