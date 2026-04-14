"""ReadFullDocument tool — reads an entire attachment and returns its text.

This is the most permissive attachment access strategy: the agent receives
the complete text of a document in a single tool call. It serves as the
baseline tool; more targeted tools (page ranges, semantic search) come later.

The tool is instantiated per agent.answer() call because the attachment path
map is per-question.
"""

from __future__ import annotations

from taxonomy_rag.readers.registry import ReaderRegistry


class ReadFullDocument:
    """Return the complete plain-text content of a named attachment."""

    name = "read_full_document"
    description = (
        "Read the complete text content of an attachment. "
        "Use this when you need to examine a document in full before forming "
        "your answer. Returns all text extracted from the file, page by page. "
        "Call list_attachments first if you are unsure which documents are available."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "attachment_name": {
                "type": "string",
                "description": (
                    "The exact name of the attachment to read, as shown in the "
                    "available attachments list (e.g. 'PERMIT-AVSP-01')."
                ),
            }
        },
        "required": ["attachment_name"],
    }

    def __init__(
        self,
        attachment_paths: dict[str, str],
        registry: ReaderRegistry,
    ) -> None:
        self._paths = attachment_paths  # {"PERMIT-AVSP-01": "/abs/path/file.pdf"}
        self._registry = registry

    def run(self, attachment_name: str) -> str:
        if attachment_name not in self._paths:
            available = ", ".join(sorted(self._paths)) or "none"
            return (
                f"Attachment '{attachment_name}' not found. "
                f"Available: {available}."
            )
        path = self._paths[attachment_name]
        return self._registry.read(path)
