"""PDF reader — delegates to PDFParser for text extraction.

PDFParser (ingestion/parsers/pdf.py) is the single fitz caller for PDFs.
PDFReader's job is to flatten the per-page list into a single plain-text
string suitable for agent tool consumption (ReadFullDocument).
"""

from __future__ import annotations


class PDFReader:
    """Extracts plain text from PDF files by delegating to PDFParser."""

    def supports(self, source: str) -> bool:
        return source.lower().endswith(".pdf")

    def read(self, source: str) -> str:
        from taxonomy_rag.ingestion.parsers.pdf import PDFParser
        doc = PDFParser().parse(source)
        return "\n\n".join(doc.pages)
