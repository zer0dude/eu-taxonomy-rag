"""PDF reader using pymupdf (fitz).

pymupdf is already a project dependency (pymupdf>=1.24 in pyproject.toml).
Each page's text is extracted and joined with a double newline. No chunking
is performed here — that is the tool layer's responsibility.
"""

from __future__ import annotations


class PDFReader:
    """Extracts plain text from PDF files page by page."""

    def supports(self, source: str) -> bool:
        return source.lower().endswith(".pdf")

    def read(self, source: str) -> str:
        import fitz  # pymupdf

        doc = fitz.open(source)
        try:
            pages = [page.get_text() for page in doc]
        finally:
            doc.close()

        return "\n\n".join(pages)
