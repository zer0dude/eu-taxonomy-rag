from functools import lru_cache

from sentence_transformers import SentenceTransformer

from taxonomy_rag.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once per process (~2-5s on first call)."""
    return SentenceTransformer(settings.embedding_model)


class Embedder:
    """Thin wrapper around SentenceTransformer.

    normalize_embeddings=True projects all vectors onto the unit sphere so that
    cosine similarity equals the dot product. This is required for consistent
    results with pgvector's <=> (cosine distance) operator.
    """

    def embed(self, text: str) -> list[float]:
        return _get_model().encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return _get_model().encode(texts, normalize_embeddings=True).tolist()
