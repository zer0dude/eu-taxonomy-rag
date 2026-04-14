"""Base types for the readers layer.

AttachmentInfo  — metadata about an attachment that agents receive upfront,
                  without needing to read the file (analogous to the header
                  information in an email attachment list).

AttachmentReader — internal Protocol that format-specific readers implement.
                   Tools use readers; agents use tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class AttachmentInfo:
    """Lightweight metadata about an attached file.

    Agents receive a list of these before any tool is called, giving them
    enough information to decide which attachments are worth reading.
    The `path` field is resolved by the eval harness and is intended for
    internal use by tools — not presented directly to the LLM.
    """

    name: str        # logical name, e.g. "PERMIT-AVSP-01"
    file_type: str   # extension without dot, e.g. "pdf", "xlsx"
    size_bytes: int  # raw file size in bytes
    path: str        # absolute resolved path on disk


@runtime_checkable
class AttachmentReader(Protocol):
    """Internal interface that format-specific readers must satisfy.

    Readers know how to extract plain text from one or more file formats.
    They have no opinion on how the text is used — that is the tool's concern.
    """

    def supports(self, source: str) -> bool:
        """Return True if this reader can handle the given source path or URL."""
        ...

    def read(self, source: str) -> str:
        """Extract and return the full plain-text content of the source."""
        ...
