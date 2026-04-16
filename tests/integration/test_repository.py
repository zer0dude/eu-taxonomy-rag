"""Integration tests for DocumentRepository — requires live PostgreSQL + pgvector.

Run with: docker compose up -d && uv run pytest tests/integration/
Tests are skipped automatically if the database is unreachable.
"""

from __future__ import annotations

import pytest


def _db_available() -> bool:
    try:
        import psycopg
        from taxonomy_rag.config import settings
        with psycopg.connect(settings.dsn, connect_timeout=2):
            return True
    except Exception:
        return False


skip_if_no_db = pytest.mark.skipif(
    not _db_available(),
    reason="PostgreSQL not reachable — run: docker compose up -d",
)


@skip_if_no_db
class TestDocumentRepository:
    """All tests share a cleanup fixture that removes inserted rows."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from taxonomy_rag.db.repository import DocumentRepository
        self.repo = DocumentRepository()
        self.inserted_ids: list[int] = []
        yield
        for doc_id in self.inserted_ids:
            self.repo.delete(doc_id)

    def _insert(self, content: str, metadata: dict | None = None) -> int:
        embedding = [0.1] * 384
        doc_id = self.repo.insert(content, embedding, metadata or {})
        self.inserted_ids.append(doc_id)
        return doc_id

    # ------------------------------------------------------------------
    # Basic CRUD
    # ------------------------------------------------------------------

    def test_insert_returns_integer_id(self):
        doc_id = self._insert("Test content.")
        assert isinstance(doc_id, int)
        assert doc_id > 0

    def test_get_by_id_roundtrip(self):
        doc_id = self._insert("Roundtrip test content.")
        doc = self.repo.get_by_id(doc_id)
        assert doc is not None
        assert doc["id"] == doc_id
        assert doc["content"] == "Roundtrip test content."

    def test_get_by_id_returns_none_for_missing(self):
        result = self.repo.get_by_id(999_999_999)
        assert result is None

    def test_delete_removes_row(self):
        doc_id = self._insert("To be deleted.")
        self.inserted_ids.remove(doc_id)  # prevent double-delete in cleanup
        deleted = self.repo.delete(doc_id)
        assert deleted is True
        assert self.repo.get_by_id(doc_id) is None

    def test_delete_returns_false_for_missing(self):
        result = self.repo.delete(999_999_999)
        assert result is False

    # ------------------------------------------------------------------
    # Metadata storage and retrieval
    # ------------------------------------------------------------------

    def test_metadata_stored_and_retrieved(self):
        meta = {"document_id": "2021_2139", "chunk_strategy": "naive", "page": 3}
        doc_id = self._insert("Metadata test.", meta)
        doc = self.repo.get_by_id(doc_id)
        assert doc["metadata"]["document_id"] == "2021_2139"
        assert doc["metadata"]["chunk_strategy"] == "naive"

    def test_metadata_filter_returns_matching_rows(self):
        tag = "integration_test_filter"
        self._insert("Doc A.", {"source": tag, "category": "A"})
        self._insert("Doc B.", {"source": tag, "category": "B"})
        self._insert("Doc C.", {"source": "other"})

        results = self.repo.get_all(metadata_filter={"source": tag})
        sources = [r["metadata"]["source"] for r in results]
        assert all(s == tag for s in sources)
        categories = {r["metadata"]["category"] for r in results}
        assert {"A", "B"} <= categories  # at least A and B present

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def test_vector_search_returns_results(self):
        self._insert("Sustainability criteria.", {"tag": "vs_test"})
        embedding = [0.1] * 384
        results = self.repo.vector_search(embedding, top_k=5)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_vector_search_results_have_score_field(self):
        self._insert("Some content.", {})
        embedding = [0.1] * 384
        results = self.repo.vector_search(embedding, top_k=1)
        assert len(results) >= 1
        assert "score" in results[0]
        assert isinstance(results[0]["score"], float)

    def test_vector_search_ordered_by_score_descending(self):
        self._insert("First doc.", {})
        self._insert("Second doc.", {})
        embedding = [0.1] * 384
        results = self.repo.vector_search(embedding, top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def test_hybrid_search_returns_list(self):
        self._insert("EU Taxonomy sustainable finance.", {"document_id": "test_hybrid"})
        embedding = [0.1] * 384
        results = self.repo.hybrid_search("taxonomy", embedding, top_k=5)
        assert isinstance(results, list)

    def test_hybrid_search_results_have_score_field(self):
        self._insert("EU Taxonomy climate criteria.", {})
        embedding = [0.1] * 384
        results = self.repo.hybrid_search("taxonomy climate", embedding, top_k=5)
        if results:
            assert "score" in results[0]
