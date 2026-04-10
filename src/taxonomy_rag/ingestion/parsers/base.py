from typing import Protocol

from taxonomy_rag.ingestion.models import ParsedDocument


class DocumentParser(Protocol):
    """Interface that all document parsers must implement.

    Add new parsers as new files in this directory — never modify existing ones.
    """

    def parse(self, file_path: str) -> ParsedDocument:
        """Parse the file at file_path and return a ParsedDocument."""
        ...

    def supports(self, file_path: str) -> bool:
        """Return True if this parser can handle the given file."""
        ...
