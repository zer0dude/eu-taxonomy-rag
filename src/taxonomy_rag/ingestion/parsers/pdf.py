"""PDF parser using pymupdf (fitz).

Extracts raw text page-by-page using fitz's default text extraction mode.
Known limitations for multi-column regulatory PDFs:
- Annex tables with multiple columns may have columns merged left-to-right
- Footnotes can interleave with body text
- Page headers/footers appear on every page

These limitations are acceptable for the naive baseline; a future parser can
use fitz's 'blocks' or 'dict' extraction modes to improve reading order.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # pymupdf

from taxonomy_rag.ingestion.models import ParsedDocument

# Maps the parent directory name to a document_type value stored in metadata.
_DIR_TYPE_MAP: dict[str, str] = {
    "core_regulation":                              "regulation",
    "delegated_acts_technical_criteria":            "delegated_act",
    "guidance_documents":                           "guidance",
    "comission_notices_interpretive_guidance_faqs": "notice",
}

# Matches the trailing language/format suffix added by EUR-Lex downloads.
_SUFFIX_RE = re.compile(r"_EN_TXT$", re.IGNORECASE)


def _extract_document_id(file_path: str) -> str:
    """Derive a clean document identifier from the filename.

    Examples:
        CELEX_32021R2139_EN_TXT.pdf  →  32021R2139
        OJ_L_202302485_EN_TXT.pdf    →  OJ_L_202302485
        Taxonomy User Guide.pdf      →  taxonomy_user_guide
    """
    stem = Path(file_path).stem                  # strip .pdf extension
    doc_id = stem.removeprefix("CELEX_")         # strip CELEX_ prefix if present
    doc_id = _SUFFIX_RE.sub("", doc_id)          # strip _EN_TXT suffix if present
    doc_id = doc_id.lower().replace(" ", "_")    # normalise whitespace
    return doc_id


class PDFParser:
    """Parses PDF files into a ParsedDocument using pymupdf.

    Text is extracted page-by-page via fitz's default plain-text mode.
    document_id and document_type are inferred from the filename and the
    parent directory respectively, matching the eu-tax-docs folder structure.
    """

    def parse(self, file_path: str) -> ParsedDocument:
        doc = fitz.open(file_path)
        pages: list[str] = [page.get_text() for page in doc]

        document_id = _extract_document_id(file_path)

        dir_name = Path(file_path).parent.name
        document_type = _DIR_TYPE_MAP.get(dir_name, "unknown")

        # Prefer the embedded PDF title; fall back to the filename stem.
        title: str = doc.metadata.get("title") or Path(file_path).stem

        return ParsedDocument(
            source_path=str(file_path),
            document_id=document_id,
            document_type=document_type,
            title=title,
            pages=pages,
            metadata={"page_count": len(pages)},
        )

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith(".pdf")
