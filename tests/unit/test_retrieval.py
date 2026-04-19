"""Unit tests for the retrieval layer: CorpusScope and NaiveVectorRetrieval."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from taxonomy_rag.retrieval.base import RetrievalResult
from taxonomy_rag.retrieval.naive import NaiveVectorRetrieval
from taxonomy_rag.retrieval.scope import CorpusScope, NAIVE_PDF_CORPUS


# ---------------------------------------------------------------------------
# CorpusScope.to_metadata_filter
# ---------------------------------------------------------------------------

class TestCorpusScope:
    def test_strategy_only(self):
        scope = CorpusScope(ingestion_strategy="naive_pdf")
        assert scope.to_metadata_filter() == {"ingestion_strategy": "naive_pdf"}

    def test_document_type_only(self):
        scope = CorpusScope(document_type="delegated_act")
        assert scope.to_metadata_filter() == {"document_type": "delegated_act"}

    def test_both_fields(self):
        scope = CorpusScope(
            ingestion_strategy="naive_pdf", document_type="delegated_act"
        )
        assert scope.to_metadata_filter() == {
            "ingestion_strategy": "naive_pdf",
            "document_type": "delegated_act",
        }

    def test_neither_field_returns_none(self):
        scope = CorpusScope()
        assert scope.to_metadata_filter() is None

    def test_naive_pdf_corpus_constant(self):
        assert NAIVE_PDF_CORPUS.ingestion_strategy == "naive_pdf"
        assert NAIVE_PDF_CORPUS.document_type is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_DOCS = [
    {
        "id": 1,
        "content": "Wind energy technical screening criteria.",
        "score": 0.92,
        "metadata": {"document_id": "32021r2139", "page_range": "45-46"},
    },
    {
        "id": 2,
        "content": "Solar PV activity description.",
        "score": 0.85,
        "metadata": {"document_id": "32021r2139", "page_range": "50"},
    },
]


def _make_repo(docs=_MOCK_DOCS):
    repo = MagicMock()
    repo.vector_search.return_value = docs
    return repo


def _make_embedder():
    emb = MagicMock()
    emb.embed.return_value = [0.1] * 384
    return emb


# ---------------------------------------------------------------------------
# NaiveVectorRetrieval.retrieve
# ---------------------------------------------------------------------------

class TestNaiveVectorRetrieval:
    def test_returns_retrieval_results(self):
        repo = _make_repo()
        emb = _make_embedder()
        retrieval = NaiveVectorRetrieval(repo=repo, embedder=emb, top_k=5)

        results = retrieval.retrieve("wind energy", scope=None)

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].doc_id == 1
        assert results[0].content == "Wind energy technical screening criteria."
        assert results[0].score == pytest.approx(0.92)
        assert results[0].metadata == {"document_id": "32021r2139", "page_range": "45-46"}

    def test_passes_embedding_to_repo(self):
        repo = _make_repo()
        emb = _make_embedder()
        retrieval = NaiveVectorRetrieval(repo=repo, embedder=emb, top_k=5)

        retrieval.retrieve("query")

        emb.embed.assert_called_once_with("query")
        repo.vector_search.assert_called_once()
        call_args = repo.vector_search.call_args
        assert call_args[0][0] == [0.1] * 384
        assert call_args[0][1] == 5

    def test_no_scope_passes_none_filter(self):
        repo = _make_repo()
        emb = _make_embedder()
        retrieval = NaiveVectorRetrieval(repo=repo, embedder=emb)

        retrieval.retrieve("query", scope=None)

        _, kwargs = repo.vector_search.call_args
        assert kwargs.get("metadata_filter") is None

    def test_scope_passes_metadata_filter(self):
        repo = _make_repo()
        emb = _make_embedder()
        scope = CorpusScope(ingestion_strategy="naive_pdf")
        retrieval = NaiveVectorRetrieval(repo=repo, embedder=emb)

        retrieval.retrieve("query", scope=scope)

        _, kwargs = repo.vector_search.call_args
        assert kwargs["metadata_filter"] == {"ingestion_strategy": "naive_pdf"}

    def test_empty_results(self):
        repo = _make_repo(docs=[])
        emb = _make_embedder()
        retrieval = NaiveVectorRetrieval(repo=repo, embedder=emb)

        results = retrieval.retrieve("query")

        assert results == []

    def test_missing_metadata_defaults_to_empty_dict(self):
        repo = MagicMock()
        repo.vector_search.return_value = [
            {"id": 1, "content": "text", "score": 0.5}
        ]
        emb = _make_embedder()
        retrieval = NaiveVectorRetrieval(repo=repo, embedder=emb)

        results = retrieval.retrieve("query")

        assert results[0].metadata == {}
