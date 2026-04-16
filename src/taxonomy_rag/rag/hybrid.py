from __future__ import annotations

from typing import Any

import litellm

from taxonomy_rag.db.repository import DocumentRepository
from taxonomy_rag.embeddings.embedder import Embedder
from taxonomy_rag.llm.provider import get_completion_kwargs

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the context provided below. "
    "If the context does not contain enough information to answer, say so clearly."
)


class HybridRAG:
    """Hybrid RAG pipeline: embed → RRF(BM25 + cosine) → generate.

    Combines two retrieval signals before generation:
    - BM25 full-text search (postgres tsvector): good for keyword matches
    - Cosine vector search (pgvector HNSW): good for semantic similarity

    The two ranked lists are merged with Reciprocal Rank Fusion (RRF) entirely
    inside a single SQL CTE — no extra round-trip to the database.

    Same public interface as NaiveRAG so the two are interchangeable.
    """

    def __init__(
        self,
        repo: DocumentRepository | None = None,
        embedder: Embedder | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.repo = repo or DocumentRepository()
        self.embedder = embedder or Embedder()
        self.rrf_k = rrf_k

    def ingest(self, content: str, metadata: dict[str, Any] | None = None) -> int:
        embedding = self.embedder.embed(content)
        return self.repo.insert(content, embedding, metadata)

    def query(
        self,
        question: str,
        top_k: int = 5,
        llm_provider: str | None = None,
    ) -> dict[str, Any]:
        embedding = self.embedder.embed(question)
        docs = self.repo.hybrid_search(question, embedding, top_k, self.rrf_k)
        context = "\n\n".join(d["content"] for d in docs)

        messages = [
            {
                "role": "system",
                "content": f"{_SYSTEM_PROMPT}\n\nContext:\n{context}",
            },
            {"role": "user", "content": question},
        ]

        kwargs = get_completion_kwargs(llm_provider)
        response = litellm.completion(**kwargs, messages=messages)
        answer = response.choices[0].message.content or ""

        return {
            "answer": answer,
            "sources": [
                {"id": d["id"], "content": d["content"], "score": float(d["score"])}
                for d in docs
            ],
        }
