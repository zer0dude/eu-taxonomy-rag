from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CorpusScope:
    """Defines which portion of the document store a retrieval tool may access.

    Both fields are optional; omitting them imposes no restriction on that axis.
    """

    name: str
    ingestion_strategy: str | None = None
    document_type: str | None = None

    def to_metadata_filter(self) -> dict | None:
        f: dict = {}
        if self.ingestion_strategy:
            f["ingestion_strategy"] = self.ingestion_strategy
        if self.document_type:
            f["document_type"] = self.document_type
        return f or None


NAIVE_PDF_CORPUS = CorpusScope(
    name="naive_pdf_corpus",
    ingestion_strategy="naive_pdf",
)
