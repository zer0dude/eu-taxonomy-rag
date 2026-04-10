from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    """Raw output of a parser — one per source file."""

    source_path: str
    document_id: str        # e.g. "2021_2139" (regulation number, underscored)
    document_type: str      # "regulation" | "delegated_act" | "corrigendum" | "spreadsheet"
    title: str
    pages: list[str]        # raw text per page, in order
    metadata: dict = field(default_factory=dict)  # any structured info extracted by the parser


@dataclass
class Chunk:
    """A single unit ready for embedding and storage."""

    content: str
    metadata: dict = field(default_factory=dict)
    # metadata must include: source, document_id, chunk_strategy
    # metadata should include where known: article, annex, page_range
