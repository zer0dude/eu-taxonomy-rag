from __future__ import annotations

from functools import lru_cache
from typing import Any

import litellm

from taxonomy_rag.db.repository import DocumentRepository
from taxonomy_rag.embeddings.embedder import Embedder
from taxonomy_rag.llm.provider import get_completion_kwargs

_HYDE_SYSTEM = (
    "Write a short, factual paragraph that would directly answer the question below. "
    "Do not hedge or say you don't know — just write what a good answer would look like."
)

_RAG_SYSTEM = (
    "You are a helpful assistant. Answer using ONLY the context provided below. "
    "If the context does not contain enough information to answer, say so clearly."
)


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load the cross-encoder once per process (~1-2s)."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class AdvancedRAG:
    """Advanced RAG with two optional enhancements over NaiveRAG:

    HyDE (Hypothetical Document Embeddings)
    ----------------------------------------
    Instead of embedding the question directly, we ask the LLM to generate a
    hypothetical answer document, then embed *that*.  Rationale: a generated
    answer sits closer in embedding space to real answers than a short question
    does, improving recall for semantic search.

    Cross-encoder reranking
    -----------------------
    After first-pass retrieval (e.g. top-20 by cosine similarity), a
    cross-encoder scores every (question, document) pair and picks the true
    top-k.  Cross-encoders are O(n) but far more accurate than bi-encoders
    because they see both query and document together.

    Both can be enabled/disabled independently.
    """

    def __init__(
        self,
        repo: DocumentRepository | None = None,
        embedder: Embedder | None = None,
        use_hyde: bool = True,
        use_reranking: bool = True,
        rerank_candidates: int = 20,
    ) -> None:
        self.repo = repo or DocumentRepository()
        self.embedder = embedder or Embedder()
        self.use_hyde = use_hyde
        self.use_reranking = use_reranking
        self.rerank_candidates = rerank_candidates

    def ingest(self, content: str, metadata: dict[str, Any] | None = None) -> int:
        embedding = self.embedder.embed(content)
        return self.repo.insert(content, embedding, metadata)

    def _generate_hypothetical_doc(self, question: str, provider: str | None) -> str:
        messages = [
            {"role": "system", "content": _HYDE_SYSTEM},
            {"role": "user", "content": question},
        ]
        kwargs = get_completion_kwargs(provider)
        response = litellm.completion(**kwargs, messages=messages)
        # Fall back to the raw question if the model returns nothing
        return response.choices[0].message.content or question

    def _rerank(self, question: str, docs: list[dict], top_k: int) -> list[dict]:
        pairs = [(question, d["content"]) for d in docs]
        scores = _get_cross_encoder().predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [{"score": float(s), **d} for s, d in ranked[:top_k]]

    def query(
        self,
        question: str,
        top_k: int = 5,
        llm_provider: str | None = None,
    ) -> dict[str, Any]:
        # Step 1: determine what to embed
        query_text = (
            self._generate_hypothetical_doc(question, llm_provider)
            if self.use_hyde
            else question
        )

        # Step 2: first-pass retrieval (fetch more candidates for reranking)
        candidate_k = self.rerank_candidates if self.use_reranking else top_k
        embedding = self.embedder.embed(query_text)
        docs = self.repo.vector_search(embedding, candidate_k)

        # Step 3: optional cross-encoder reranking
        if self.use_reranking and len(docs) > top_k:
            docs = self._rerank(question, docs, top_k)
        else:
            docs = docs[:top_k]

        # Step 4: generate answer
        context = "\n\n".join(d["content"] for d in docs)
        messages = [
            {
                "role": "system",
                "content": f"{_RAG_SYSTEM}\n\nContext:\n{context}",
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
