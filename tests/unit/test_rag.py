"""Unit tests for NaiveRAG and HybridRAG — mocked DB and litellm."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


MOCK_DOCS = [
    {"id": 1, "content": "Sustainable activities text.", "score": 0.95},
    {"id": 2, "content": "Taxonomy regulation overview.", "score": 0.88},
]


def _litellm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_repo(docs=MOCK_DOCS):
    repo = MagicMock()
    repo.vector_search.return_value = docs
    repo.hybrid_search.return_value = docs
    return repo


def _make_embedder():
    emb = MagicMock()
    emb.embed.return_value = [0.1] * 384
    return emb


# ---------------------------------------------------------------------------
# NaiveRAG
# ---------------------------------------------------------------------------

class TestNaiveRAG:
    def _rag(self, docs=MOCK_DOCS):
        from taxonomy_rag.rag.naive import NaiveRAG
        return NaiveRAG(repo=_make_repo(docs), embedder=_make_embedder())

    def test_query_returns_answer_and_sources(self):
        rag = self._rag()
        with patch("taxonomy_rag.rag.naive.litellm.completion", return_value=_litellm_response("The answer.")):
            with patch("taxonomy_rag.rag.naive.get_completion_kwargs", return_value={"model": "test"}):
                result = rag.query("What is sustainability?")

        assert result["answer"] == "The answer."
        assert len(result["sources"]) == 2
        assert result["sources"][0]["id"] == 1
        assert result["sources"][0]["score"] == pytest.approx(0.95)

    def test_query_builds_system_message_with_context(self):
        rag = self._rag()
        captured: dict = {}

        def capture(**kwargs):
            captured["messages"] = kwargs["messages"]
            return _litellm_response("ok")

        with patch("taxonomy_rag.rag.naive.litellm.completion", side_effect=capture):
            with patch("taxonomy_rag.rag.naive.get_completion_kwargs", return_value={"model": "test"}):
                rag.query("What?")

        system_msg = captured["messages"][0]
        assert system_msg["role"] == "system"
        assert "Sustainable activities text." in system_msg["content"]
        assert "Taxonomy regulation overview." in system_msg["content"]

    def test_query_passes_question_as_user_message(self):
        rag = self._rag(docs=[])
        captured: dict = {}

        def capture(**kwargs):
            captured["messages"] = kwargs["messages"]
            return _litellm_response("ok")

        with patch("taxonomy_rag.rag.naive.litellm.completion", side_effect=capture):
            with patch("taxonomy_rag.rag.naive.get_completion_kwargs", return_value={"model": "test"}):
                rag.query("Specific question?")

        user_msg = captured["messages"][1]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "Specific question?"

    def test_empty_sources_returns_empty_list(self):
        rag = self._rag(docs=[])
        with patch("taxonomy_rag.rag.naive.litellm.completion", return_value=_litellm_response("no info")):
            with patch("taxonomy_rag.rag.naive.get_completion_kwargs", return_value={"model": "test"}):
                result = rag.query("Q?")

        assert result["sources"] == []

    def test_ingest_embeds_and_inserts(self):
        from taxonomy_rag.rag.naive import NaiveRAG
        repo = _make_repo()
        repo.insert.return_value = 42
        embedder = _make_embedder()
        rag = NaiveRAG(repo=repo, embedder=embedder)

        doc_id = rag.ingest("Some content.", {"tag": "test"})

        embedder.embed.assert_called_once_with("Some content.")
        repo.insert.assert_called_once()
        assert doc_id == 42


# ---------------------------------------------------------------------------
# HybridRAG
# ---------------------------------------------------------------------------

class TestHybridRAG:
    def _rag(self, docs=MOCK_DOCS):
        from taxonomy_rag.rag.hybrid import HybridRAG
        return HybridRAG(repo=_make_repo(docs), embedder=_make_embedder())

    def test_query_calls_hybrid_search_not_vector_search(self):
        from taxonomy_rag.rag.hybrid import HybridRAG
        repo = _make_repo()
        rag = HybridRAG(repo=repo, embedder=_make_embedder())

        with patch("taxonomy_rag.rag.hybrid.litellm.completion", return_value=_litellm_response("answer")):
            with patch("taxonomy_rag.rag.hybrid.get_completion_kwargs", return_value={"model": "test"}):
                rag.query("question")

        repo.hybrid_search.assert_called_once()
        repo.vector_search.assert_not_called()

    def test_query_returns_answer(self):
        rag = self._rag()
        with patch("taxonomy_rag.rag.hybrid.litellm.completion", return_value=_litellm_response("hybrid answer")):
            with patch("taxonomy_rag.rag.hybrid.get_completion_kwargs", return_value={"model": "test"}):
                result = rag.query("question")

        assert result["answer"] == "hybrid answer"
        assert len(result["sources"]) == 2

    def test_system_message_contains_context(self):
        rag = self._rag()
        captured: dict = {}

        def capture(**kwargs):
            captured["messages"] = kwargs["messages"]
            return _litellm_response("ok")

        with patch("taxonomy_rag.rag.hybrid.litellm.completion", side_effect=capture):
            with patch("taxonomy_rag.rag.hybrid.get_completion_kwargs", return_value={"model": "test"}):
                rag.query("Q?")

        system_content = captured["messages"][0]["content"]
        assert "Sustainable activities text." in system_content
