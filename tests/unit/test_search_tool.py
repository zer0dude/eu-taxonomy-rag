"""Unit tests for SearchCorpusTool."""

from __future__ import annotations

from unittest.mock import MagicMock

from taxonomy_rag.retrieval.base import RetrievalResult
from taxonomy_rag.retrieval.scope import CorpusScope, NAIVE_PDF_CORPUS
from taxonomy_rag.tools.search.corpus import SearchCorpusTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retrieval(results: list[RetrievalResult] | None = None):
    mock = MagicMock()
    mock.retrieve.return_value = results if results is not None else []
    return mock


def _make_tool(results=None):
    retrieval = _make_retrieval(results)
    scope = NAIVE_PDF_CORPUS
    return SearchCorpusTool(retrieval, scope), retrieval


# ---------------------------------------------------------------------------
# Tool protocol compliance
# ---------------------------------------------------------------------------

class TestSearchCorpusToolProtocol:
    def test_name(self):
        tool, _ = _make_tool()
        assert tool.name == "search_corpus"

    def test_description_is_nonempty_string(self):
        tool, _ = _make_tool()
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_input_schema_structure(self):
        tool, _ = _make_tool()
        schema = tool.input_schema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_run_returns_string(self):
        tool, _ = _make_tool(results=[])
        assert isinstance(tool.run(query="test"), str)


# ---------------------------------------------------------------------------
# SearchCorpusTool.run — formatting
# ---------------------------------------------------------------------------

class TestSearchCorpusToolRun:
    def test_empty_results_message(self):
        tool, _ = _make_tool(results=[])
        assert tool.run(query="anything") == "No relevant documents found."

    def test_formats_document_id_and_score(self):
        results = [
            RetrievalResult(
                doc_id=1,
                content="Technical screening text.",
                score=0.92,
                metadata={"document_id": "32021r2139", "page_range": "45-46"},
            )
        ]
        tool, _ = _make_tool(results=results)
        output = tool.run(query="wind energy")

        assert "[1]" in output
        assert "32021r2139" in output
        assert "45-46" in output
        assert "0.920" in output
        assert "Technical screening text." in output

    def test_omits_page_range_when_missing(self):
        results = [
            RetrievalResult(
                doc_id=1,
                content="Some text.",
                score=0.7,
                metadata={"document_id": "32021r2139"},
            )
        ]
        tool, _ = _make_tool(results=results)
        output = tool.run(query="query")

        assert "Pages:" not in output
        assert "32021r2139" in output

    def test_multiple_results_numbered(self):
        results = [
            RetrievalResult(doc_id=1, content="First.", score=0.9, metadata={"document_id": "doc1"}),
            RetrievalResult(doc_id=2, content="Second.", score=0.8, metadata={"document_id": "doc2"}),
            RetrievalResult(doc_id=3, content="Third.", score=0.7, metadata={"document_id": "doc3"}),
        ]
        tool, _ = _make_tool(results=results)
        output = tool.run(query="query")

        assert "[1]" in output
        assert "[2]" in output
        assert "[3]" in output
        assert "First." in output
        assert "Second." in output
        assert "Third." in output

    def test_passes_query_and_scope_to_retrieval(self):
        tool, retrieval = _make_tool(results=[])
        scope = CorpusScope(ingestion_strategy="naive_pdf")
        tool._scope = scope

        tool.run(query="solar criteria")

        retrieval.retrieve.assert_called_once_with("solar criteria", scope)

    def test_unknown_document_id_shows_unknown(self):
        results = [
            RetrievalResult(doc_id=1, content="Text.", score=0.5, metadata={})
        ]
        tool, _ = _make_tool(results=results)
        output = tool.run(query="query")
        assert "unknown" in output
