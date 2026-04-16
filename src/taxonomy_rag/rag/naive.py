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


class NaiveRAG:
    """Naive RAG pipeline: embed → vector search → generate.

    This is the simplest possible RAG implementation:
    1. Embed the question with sentence-transformers
    2. Retrieve top-k documents by cosine similarity
    3. Concatenate their content as context
    4. Pass context + question to the LLM via litellm.completion()
    """

    def __init__(
        self,
        repo: DocumentRepository | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.repo = repo or DocumentRepository()
        self.embedder = embedder or Embedder()

    def ingest(self, content: str, metadata: dict[str, Any] | None = None) -> int:
        """Embed content and store it. Returns the new document id."""
        embedding = self.embedder.embed(content)
        return self.repo.insert(content, embedding, metadata)

    def query(
        self,
        question: str,
        top_k: int = 5,
        llm_provider: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant documents and generate an answer.

        Returns:
            {
                "answer": str,
                "sources": [{"id": int, "content": str, "score": float}, ...]
            }
        """
        embedding = self.embedder.embed(question)
        docs = self.repo.vector_search(embedding, top_k)
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
