from __future__ import annotations

from functools import lru_cache

import litellm

from taxonomy_rag.db.repository import DocumentRepository
from taxonomy_rag.embeddings.embedder import Embedder
from taxonomy_rag.llm.provider import get_completion_kwargs
from taxonomy_rag.retrieval.base import RetrievalResult
from taxonomy_rag.retrieval.scope import CorpusScope
from taxonomy_rag.tracing.base import NullTracer, Tracer

_HYDE_SYSTEM = (
    "Write a short, factual paragraph that would directly answer the question below. "
    "Do not hedge or say you don't know — just write what a good answer would look like."
)

# Iteration index reserved for HyDE pre-retrieval LLM calls.
# AgentLoop starts at iteration 1, so 0 is safe and unambiguous in traces.
_HYDE_ITERATION = 0


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load the cross-encoder once per process (~1-2s on first call)."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class AdvancedRetrieval:
    """Retrieval: HyDE query expansion + cosine vector search + cross-encoder reranking.

    HyDE (Hypothetical Document Embeddings)
    ----------------------------------------
    Instead of embedding the question directly, an LLM generates a hypothetical
    answer paragraph, which is then embedded. Generated answers sit closer in
    embedding space to real answers than short questions do, improving recall.
    HyDE tokens are logged to the tracer at iteration index 0 (pre-loop).

    Cross-encoder reranking
    -----------------------
    After first-pass retrieval (top rerank_candidates by cosine), a cross-encoder
    scores every (question, document) pair and selects the true top_k. More
    accurate than bi-encoder cosine alone; O(rerank_candidates) at query time.

    Both enhancements can be disabled independently for ablation experiments.

    Compose with a CorpusScope and wrap in SearchCorpusTool to expose as an agent tool.
    """

    def __init__(
        self,
        repo: DocumentRepository | None = None,
        embedder: Embedder | None = None,
        top_k: int = 5,
        use_hyde: bool = True,
        use_reranking: bool = True,
        rerank_candidates: int = 20,
    ) -> None:
        self.repo = repo or DocumentRepository()
        self.embedder = embedder or Embedder()
        self.top_k = top_k
        self.use_hyde = use_hyde
        self.use_reranking = use_reranking
        self.rerank_candidates = rerank_candidates

    def _generate_hypothetical_doc(self, question: str, tracer: Tracer) -> str:
        messages = [
            {"role": "system", "content": _HYDE_SYSTEM},
            {"role": "user", "content": question},
        ]
        kwargs = get_completion_kwargs()
        response = litellm.completion(**kwargs, messages=messages, max_tokens=256)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        tracer.log_usage(_HYDE_ITERATION, input_tokens, output_tokens)

        return response.choices[0].message.content or question

    def _rerank(self, question: str, docs: list[dict], top_k: int) -> list[dict]:
        pairs = [(question, d["content"]) for d in docs]
        scores = _get_cross_encoder().predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [{"score": float(s), **d} for s, d in ranked[:top_k]]

    def retrieve(
        self,
        query: str,
        scope: CorpusScope | None = None,
        tracer: Tracer = NullTracer(),
    ) -> list[RetrievalResult]:
        query_text = (
            self._generate_hypothetical_doc(query, tracer) if self.use_hyde else query
        )

        embedding = self.embedder.embed(query_text)
        metadata_filter = scope.to_metadata_filter() if scope else None
        candidate_k = self.rerank_candidates if self.use_reranking else self.top_k
        docs = self.repo.vector_search(embedding, candidate_k, metadata_filter=metadata_filter)

        if self.use_reranking and len(docs) > self.top_k:
            docs = self._rerank(query, docs, self.top_k)
        else:
            docs = docs[: self.top_k]

        return [
            RetrievalResult(
                doc_id=d["id"],
                content=d["content"],
                score=float(d["score"]),
                metadata=d.get("metadata") or {},
            )
            for d in docs
        ]
