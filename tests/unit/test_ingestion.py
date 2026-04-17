"""Unit tests for PDFParser and NaiveChunker.

All tests are pure — no DB, no embedding model, no real PDF files.
fitz.open() is mocked so tests run without a real PDF on disk.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from taxonomy_rag.ingestion.chunkers.naive import NaiveChunker
from taxonomy_rag.ingestion.models import ParsedDocument
from taxonomy_rag.ingestion.parsers.pdf import PDFParser, _extract_document_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(pages: list[str], page_count: int | None = None) -> ParsedDocument:
    """Build a minimal ParsedDocument for chunker tests."""
    return ParsedDocument(
        source_path="/tmp/test.pdf",
        document_id="test_001",
        document_type="regulation",
        title="Test",
        pages=pages,
        metadata={"page_count": page_count or len(pages)},
    )


def _mock_fitz_doc(pages: list[str], title: str = "Test Title") -> MagicMock:
    """Return a MagicMock that behaves like a fitz.Document."""
    mock_doc = MagicMock()
    mock_doc.metadata = {"title": title}
    mock_doc.__iter__ = MagicMock(
        return_value=iter([MagicMock(get_text=MagicMock(return_value=p)) for p in pages])
    )
    mock_doc.__len__ = MagicMock(return_value=len(pages))
    return mock_doc


# ---------------------------------------------------------------------------
# PDFParser — supports()
# ---------------------------------------------------------------------------

class TestPDFParserSupports:
    def test_supports_lowercase_pdf(self):
        assert PDFParser().supports("document.pdf") is True

    def test_supports_uppercase_pdf(self):
        assert PDFParser().supports("DOCUMENT.PDF") is True

    def test_does_not_support_xlsx(self):
        assert PDFParser().supports("spreadsheet.xlsx") is False

    def test_does_not_support_txt(self):
        assert PDFParser().supports("notes.txt") is False

    def test_does_not_support_no_extension(self):
        assert PDFParser().supports("noextension") is False


# ---------------------------------------------------------------------------
# PDFParser — document_id extraction (via _extract_document_id helper)
# ---------------------------------------------------------------------------

class TestDocumentIdExtraction:
    def test_celex_with_en_txt_suffix(self):
        assert _extract_document_id("CELEX_32021R2139_EN_TXT.pdf") == "32021r2139"

    def test_oj_identifier(self):
        assert _extract_document_id("OJ_L_202302485_EN_TXT.pdf") == "oj_l_202302485"

    def test_user_guide_spaces_normalised(self):
        result = _extract_document_id("Taxonomy User Guide.pdf")
        assert result == "taxonomy_user_guide"

    def test_bare_celex_no_suffix(self):
        # File without _EN_TXT should still strip CELEX_ prefix
        assert _extract_document_id("CELEX_32020R0852.pdf") == "32020r0852"


# ---------------------------------------------------------------------------
# PDFParser — parse()
# ---------------------------------------------------------------------------

class TestPDFParserParse:
    def _run_parse(self, file_path: str, pages: list[str], title: str = "PDF Title") -> ParsedDocument:
        mock_doc = _mock_fitz_doc(pages, title=title)
        with patch("taxonomy_rag.ingestion.parsers.pdf.fitz.open", return_value=mock_doc):
            return PDFParser().parse(file_path)

    def test_returns_parsed_document_type(self):
        result = self._run_parse("/some/dir/file.pdf", ["page one"])
        assert isinstance(result, ParsedDocument)

    def test_pages_extracted(self):
        pages = ["first page text", "second page text", "third page text"]
        result = self._run_parse("/some/dir/file.pdf", pages)
        assert result.pages == pages

    def test_source_path_preserved(self):
        path = "/data/raw/eu-tax-docs/core_regulation/CELEX_32020R0852_EN_TXT.pdf"
        result = self._run_parse(path, ["text"])
        assert result.source_path == path

    def test_title_from_pdf_metadata(self):
        result = self._run_parse("/some/dir/file.pdf", ["text"], title="Official Title")
        assert result.title == "Official Title"

    def test_title_fallback_to_stem_when_empty(self):
        mock_doc = MagicMock()
        mock_doc.metadata = {"title": ""}  # empty title
        mock_doc.__iter__ = MagicMock(return_value=iter([MagicMock(get_text=MagicMock(return_value="text"))]))
        with patch("taxonomy_rag.ingestion.parsers.pdf.fitz.open", return_value=mock_doc):
            result = PDFParser().parse("/some/path/my_file_name.pdf")
        assert result.title == "my_file_name"

    def test_page_count_in_metadata(self):
        result = self._run_parse("/some/dir/file.pdf", ["a", "b", "c"])
        assert result.metadata["page_count"] == 3

    def test_celex_document_id_extraction(self):
        result = self._run_parse(
            "/data/delegated_acts_technical_criteria/CELEX_32021R2139_EN_TXT.pdf",
            ["text"],
        )
        assert result.document_id == "32021r2139"

    def test_oj_document_id_extraction(self):
        result = self._run_parse(
            "/data/delegated_acts_technical_criteria/OJ_L_202302485_EN_TXT.pdf",
            ["text"],
        )
        assert result.document_id == "oj_l_202302485"

    def test_document_type_regulation(self):
        result = self._run_parse(
            "/data/core_regulation/CELEX_32020R0852_EN_TXT.pdf", ["text"]
        )
        assert result.document_type == "regulation"

    def test_document_type_delegated_act(self):
        result = self._run_parse(
            "/data/delegated_acts_technical_criteria/CELEX_32021R2139_EN_TXT.pdf",
            ["text"],
        )
        assert result.document_type == "delegated_act"

    def test_document_type_guidance(self):
        result = self._run_parse(
            "/data/guidance_documents/Taxonomy User Guide.pdf", ["text"]
        )
        assert result.document_type == "guidance"

    def test_document_type_notice(self):
        result = self._run_parse(
            "/data/comission_notices_interpretive_guidance_faqs/OJ_C_202300267_EN_TXT.pdf",
            ["text"],
        )
        assert result.document_type == "notice"

    def test_document_type_unknown_for_unexpected_dir(self):
        result = self._run_parse("/data/some_other_folder/file.pdf", ["text"])
        assert result.document_type == "unknown"


# ---------------------------------------------------------------------------
# NaiveChunker — basic behaviour
# ---------------------------------------------------------------------------

class TestNaiveChunkerBasic:
    def test_empty_pages_returns_empty_list(self):
        doc = _make_doc(pages=[])
        chunks = NaiveChunker().chunk(doc)
        assert chunks == []

    def test_blank_pages_returns_empty_list(self):
        doc = _make_doc(pages=["   ", "\n\n", ""])
        chunks = NaiveChunker().chunk(doc)
        assert chunks == []

    def test_short_document_produces_single_chunk(self):
        doc = _make_doc(pages=["only a few words here"])
        chunks = NaiveChunker(chunk_size=512, chunk_overlap=50).chunk(doc)
        assert len(chunks) == 1

    def test_single_chunk_contains_all_words(self):
        doc = _make_doc(pages=["alpha beta gamma delta"])
        chunks = NaiveChunker(chunk_size=512).chunk(doc)
        assert chunks[0].content == "alpha beta gamma delta"

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            NaiveChunker(chunk_size=10, chunk_overlap=10)


# ---------------------------------------------------------------------------
# NaiveChunker — chunk count
# ---------------------------------------------------------------------------

class TestNaiveChunkerCount:
    def test_exact_chunk_count(self):
        # 10 words, chunk_size=5, overlap=1 → step=4 → windows at 0,4,8 → 3 chunks
        words = " ".join(f"w{i}" for i in range(10))
        doc = _make_doc(pages=[words])
        chunks = NaiveChunker(chunk_size=5, chunk_overlap=1).chunk(doc)
        assert len(chunks) == 3

    def test_partial_last_window_included(self):
        # 11 words, chunk_size=5, overlap=1 → step=4 → windows at 0,4,8 → 3 chunks
        # (window at 8 has 3 words — should still be included)
        words = " ".join(f"w{i}" for i in range(11))
        doc = _make_doc(pages=[words])
        chunks = NaiveChunker(chunk_size=5, chunk_overlap=1).chunk(doc)
        assert len(chunks) == 3
        assert chunks[-1].content == "w8 w9 w10"


# ---------------------------------------------------------------------------
# NaiveChunker — overlap correctness
# ---------------------------------------------------------------------------

class TestNaiveChunkerOverlap:
    def test_overlap_words_shared(self):
        # chunk_size=4, overlap=2, step=2
        # words: a b c d e f g h
        # chunk 0: a b c d
        # chunk 1: c d e f   ← first 2 words are last 2 of chunk 0
        # chunk 2: e f g h
        words = "a b c d e f g h"
        doc = _make_doc(pages=[words])
        chunks = NaiveChunker(chunk_size=4, chunk_overlap=2).chunk(doc)
        assert len(chunks) == 4
        tail_of_0 = chunks[0].content.split()[-2:]
        head_of_1 = chunks[1].content.split()[:2]
        assert tail_of_0 == head_of_1

    def test_overlap_across_page_boundary(self):
        # Words on different pages — chunker should cross page boundaries seamlessly
        doc = _make_doc(pages=["a b c", "d e f"])
        chunks = NaiveChunker(chunk_size=4, chunk_overlap=2).chunk(doc)
        # Tokens: a b c d e f (6 words), step=2
        # chunk 0: a b c d  (pages 1-2)
        # chunk 1: c d e f  (pages 1-2 or 2)
        assert len(chunks) >= 2
        # overlap check
        tail_of_0 = chunks[0].content.split()[-2:]
        head_of_1 = chunks[1].content.split()[:2]
        assert tail_of_0 == head_of_1


# ---------------------------------------------------------------------------
# NaiveChunker — metadata
# ---------------------------------------------------------------------------

class TestNaiveChunkerMetadata:
    REQUIRED_KEYS = {"source", "document_id", "document_type", "chunk_strategy", "chunk_index", "page_range"}

    def _chunks(self, pages: list[str], chunk_size: int = 5, overlap: int = 1) -> list:
        doc = _make_doc(pages=pages)
        return NaiveChunker(chunk_size=chunk_size, chunk_overlap=overlap).chunk(doc)

    def test_all_required_keys_present(self):
        chunks = self._chunks(["word " * 20])
        for chunk in chunks:
            assert self.REQUIRED_KEYS.issubset(chunk.metadata.keys()), (
                f"Missing keys: {self.REQUIRED_KEYS - chunk.metadata.keys()}"
            )

    def test_chunk_strategy_is_naive(self):
        chunks = self._chunks(["word " * 20])
        assert all(c.metadata["chunk_strategy"] == "naive" for c in chunks)

    def test_document_id_propagated(self):
        chunks = self._chunks(["word " * 20])
        assert all(c.metadata["document_id"] == "test_001" for c in chunks)

    def test_document_type_propagated(self):
        chunks = self._chunks(["word " * 20])
        assert all(c.metadata["document_type"] == "regulation" for c in chunks)

    def test_source_propagated(self):
        chunks = self._chunks(["word " * 20])
        assert all(c.metadata["source"] == "/tmp/test.pdf" for c in chunks)

    def test_chunk_index_sequential(self):
        chunks = self._chunks(["word " * 30])
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_page_range_single_page(self):
        doc = _make_doc(pages=["a b c d e f g h"])
        chunks = NaiveChunker(chunk_size=4, chunk_overlap=1).chunk(doc)
        # All words are on page 1
        assert all("p1" in c.metadata["page_range"] for c in chunks)

    def test_page_range_multi_page(self):
        # With a small chunk_size, a chunk spanning pages 1 and 2 should show "p1-p2"
        doc = _make_doc(pages=["a b c", "d e f"])
        chunks = NaiveChunker(chunk_size=4, chunk_overlap=1).chunk(doc)
        page_ranges = {c.metadata["page_range"] for c in chunks}
        assert "p1-p2" in page_ranges
