"""Unit tests for the ingestion strategy layer.

Covers IngestionStrategy Protocol implementations, StrategyRegistry dispatch,
and the pipeline metadata injection wired through NaivePDFStrategy.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from taxonomy_rag.ingestion.strategies.naive_pdf import NaivePDFStrategy
from taxonomy_rag.ingestion.strategies.registry import DEFAULT_REGISTRY, StrategyRegistry


# ---------------------------------------------------------------------------
# NaivePDFStrategy
# ---------------------------------------------------------------------------

class TestNaivePDFStrategy:
    def setup_method(self):
        self.strategy = NaivePDFStrategy()

    def test_name(self):
        assert self.strategy.name == "naive_pdf"

    def test_description_non_empty(self):
        assert self.strategy.description

    def test_supports_pdf_lowercase(self):
        assert self.strategy.supports("document.pdf") is True

    def test_supports_pdf_uppercase(self):
        assert self.strategy.supports("DOCUMENT.PDF") is True

    def test_does_not_support_xlsx(self):
        assert self.strategy.supports("data.xlsx") is False

    def test_does_not_support_txt(self):
        assert self.strategy.supports("notes.txt") is False

    def test_build_pipeline_returns_pipeline(self):
        from taxonomy_rag.ingestion.pipeline import IngestionPipeline
        # build_pipeline creates real objects — just check the type and strategy_name
        pipeline = self.strategy.build_pipeline()
        assert isinstance(pipeline, IngestionPipeline)
        assert pipeline.strategy_name == "naive_pdf"


# ---------------------------------------------------------------------------
# StrategyRegistry
# ---------------------------------------------------------------------------

class TestStrategyRegistry:
    def _make_strategy(self, name: str, supports_ext: str = ".pdf"):
        s = MagicMock()
        s.name = name
        s.supports = MagicMock(side_effect=lambda fp: fp.lower().endswith(supports_ext))
        return s

    def test_get_known_strategy(self):
        s = self._make_strategy("test_strat")
        registry = StrategyRegistry([s])
        assert registry.get("test_strat") is s

    def test_get_unknown_strategy_raises(self):
        registry = StrategyRegistry([self._make_strategy("a")])
        with pytest.raises(ValueError, match="Unknown strategy"):
            registry.get("nonexistent")

    def test_names_returns_all(self):
        strategies = [self._make_strategy("a"), self._make_strategy("b")]
        registry = StrategyRegistry(strategies)
        assert set(registry.names()) == {"a", "b"}

    def test_find_for_file_matches(self):
        s = self._make_strategy("pdf_strat", ".pdf")
        registry = StrategyRegistry([s])
        assert registry.find_for_file("doc.pdf") is s

    def test_find_for_file_no_match_returns_none(self):
        s = self._make_strategy("pdf_strat", ".pdf")
        registry = StrategyRegistry([s])
        assert registry.find_for_file("doc.docx") is None

    def test_find_for_file_picks_first_match(self):
        s1 = self._make_strategy("first", ".pdf")
        s2 = self._make_strategy("second", ".pdf")
        registry = StrategyRegistry([s1, s2])
        assert registry.find_for_file("doc.pdf") is s1


# ---------------------------------------------------------------------------
# DEFAULT_REGISTRY sanity checks
# ---------------------------------------------------------------------------

class TestDefaultRegistry:
    def test_naive_pdf_registered(self):
        assert "naive_pdf" in DEFAULT_REGISTRY.names()

    def test_get_naive_pdf(self):
        s = DEFAULT_REGISTRY.get("naive_pdf")
        assert s.name == "naive_pdf"

    def test_find_pdf_file(self):
        s = DEFAULT_REGISTRY.find_for_file("some/path/doc.pdf")
        assert s is not None
        assert s.name == "naive_pdf"

    def test_no_match_for_unknown_extension(self):
        assert DEFAULT_REGISTRY.find_for_file("doc.docx") is None


# ---------------------------------------------------------------------------
# Pipeline metadata injection (via NaivePDFStrategy)
# ---------------------------------------------------------------------------

class TestPipelineMetadataInjection:
    """Verify that strategy_name and ingest_run_id reach chunk metadata."""

    def _run_pipeline(self, strategy_name: str, ingest_run_id: str | None):
        from taxonomy_rag.ingestion.pipeline import IngestionPipeline
        from taxonomy_rag.ingestion.models import ParsedDocument, Chunk

        mock_parser = MagicMock()
        mock_parser.parse.return_value = ParsedDocument(
            source_path="/tmp/doc.pdf",
            document_id="doc_001",
            document_type="regulation",
            title="Test",
            pages=["word1 word2 word3"],
        )

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [
            Chunk(content="word1 word2", metadata={"source": "/tmp/doc.pdf", "document_id": "doc_001"}),
            Chunk(content="word2 word3", metadata={"source": "/tmp/doc.pdf", "document_id": "doc_001"}),
        ]

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 384, [0.2] * 384]

        mock_repo = MagicMock()
        mock_repo.insert.return_value = 1

        pipeline = IngestionPipeline(
            parser=mock_parser,
            chunker=mock_chunker,
            embedder=mock_embedder,
            repo=mock_repo,
            strategy_name=strategy_name,
        )
        result = pipeline.run("/tmp/doc.pdf", ingest_run_id=ingest_run_id)

        # Extract the metadata dicts actually passed to repo.insert
        calls = mock_repo.insert.call_args_list
        return result, [call.kwargs.get("metadata") or call.args[2] for call in calls]

    def test_ingestion_strategy_injected(self):
        _, metadatas = self._run_pipeline("naive_pdf", ingest_run_id=None)
        assert all(m["ingestion_strategy"] == "naive_pdf" for m in metadatas)

    def test_ingest_run_id_injected(self):
        run_id = "test-run-123"
        _, metadatas = self._run_pipeline("naive_pdf", ingest_run_id=run_id)
        assert all(m["ingest_run_id"] == run_id for m in metadatas)

    def test_no_strategy_name_no_field(self):
        """Backwards-compatible: no strategy_name → field absent from metadata."""
        from taxonomy_rag.ingestion.pipeline import IngestionPipeline
        from taxonomy_rag.ingestion.models import ParsedDocument, Chunk

        mock_parser = MagicMock()
        mock_parser.parse.return_value = ParsedDocument(
            source_path="/tmp/doc.pdf", document_id="x", document_type="regulation",
            title="T", pages=["a b c"],
        )
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [
            Chunk(content="a b", metadata={"source": "/tmp/doc.pdf", "document_id": "x"}),
        ]
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.0] * 384]
        mock_repo = MagicMock()
        mock_repo.insert.return_value = 1

        pipeline = IngestionPipeline(
            parser=mock_parser, chunker=mock_chunker,
            embedder=mock_embedder, repo=mock_repo,
            strategy_name=None,
        )
        pipeline.run("/tmp/doc.pdf")

        call = mock_repo.insert.call_args
        metadata = call.kwargs.get("metadata") or call.args[2]
        assert "ingestion_strategy" not in metadata
        assert "ingest_run_id" not in metadata
