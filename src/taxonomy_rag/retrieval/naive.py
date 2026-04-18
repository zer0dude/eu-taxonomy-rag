from __future__ import annotations

from taxonomy_rag.db.repository import DocumentRepository
from taxonomy_rag.embeddings.embedder import Embedder
from taxonomy_rag.retrieval.base import RetrievalResult
from taxonomy_rag.retrieval.scope import CorpusScope


class NaiveVectorRetrieval:
    """Retrieval primitive: embed query → cosine vector search → RetrievalResult list.

    This class is purely retrieval — no LLM call, no answer generation.
    Compose with a CorpusScope and wrap in SearchCorpusTool to expose as an agent tool.
    """

    def __init__(
        self,
        repo: DocumentRepository | None = None,
        embedder: Embedder | None = None,
        top_k: int = 5,
    ) -> None:
        self.repo = repo or DocumentRepository()
        self.embedder = embedder or Embedder()
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        scope: CorpusScope | None = None,
    ) -> list[RetrievalResult]:
        embedding = self.embedder.embed(query)
        metadata_filter = scope.to_metadata_filter() if scope else None
        docs = self.repo.vector_search(embedding, self.top_k, metadata_filter=metadata_filter)
        return [
            RetrievalResult(
                doc_id=d["id"],
                content=d["content"],
                score=float(d["score"]),
                metadata=d.get("metadata") or {},
            )
            for d in docs
        ]
