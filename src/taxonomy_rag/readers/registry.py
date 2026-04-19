"""Reader registry — dispatches a source path to the correct reader.

Usage:
    registry = default_registry()
    text = registry.read("/path/to/file.pdf")

To add a new format, implement AttachmentReader and add an instance to the
list passed to ReaderRegistry (or to the list in default_registry()).
"""

from __future__ import annotations

from taxonomy_rag.readers.base import AttachmentReader
from taxonomy_rag.readers.pdf import PDFReader


class ReaderRegistry:
    """Picks the first reader whose supports() returns True for a given source."""

    def __init__(self, readers: list[AttachmentReader]) -> None:
        self._readers = readers

    def read(self, source: str) -> str:
        for reader in self._readers:
            if reader.supports(source):
                return reader.read(source)
        raise ValueError(
            f"No reader supports source: {source!r}. "
            f"Registered readers: {[type(r).__name__ for r in self._readers]}"
        )


def default_registry() -> ReaderRegistry:
    """Return a registry pre-loaded with all available readers."""
    return ReaderRegistry([PDFReader()])
