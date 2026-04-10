"""PDF parser using pymupdf (fitz).

STUB — implement this file to enable PDF ingestion.

Suggested approach:
    import fitz  # pymupdf
    doc = fitz.open(file_path)
    pages = [page.get_text() for page in doc]

For multi-column regulatory PDFs consider fitz's 'blocks' or 'dict' extraction
modes to reconstruct reading order correctly.
"""

from taxonomy_rag.ingestion.models import ParsedDocument


class PDFParser:
    """Parses PDF files into a ParsedDocument.

    Implement parse() using pymupdf. Pay attention to:
    - Multi-column table layouts in annexes
    - Recital / article / annex section boundaries
    - Corrigendum documents that amend earlier texts
    """

    def parse(self, file_path: str) -> ParsedDocument:
        raise NotImplementedError(
            "PDFParser.parse: implement PDF text extraction here using pymupdf (fitz). "
            "See module docstring for a starting point."
        )

    def supports(self, file_path: str) -> bool:
        raise NotImplementedError(
            "PDFParser.supports: return True if file_path ends with '.pdf' (case-insensitive)."
        )
