"""Metadata-scoped corpus search tools.

STUB — implement run() in each class to enable scoped retrieval.

These tools let the agent narrow its search to a specific source document
or annex before running a semantic search, reducing noise in retrieval.
They conform to the Tool protocol in tools/base.py.
"""

from __future__ import annotations


class FilterByDocument:
    """Search within a specific source document only.

    Useful when the agent has already identified the relevant regulation
    (e.g. '2021_2139') and wants to search within it exclusively.
    """

    name = "filter_by_document"
    description = (
        "Search within a specific source document only. "
        "Use this to scope retrieval to a single regulation or delegated act "
        "when you know which document contains the answer."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": (
                    "The document identifier, e.g. '2021_2139' or '2020_852'."
                ),
            },
            "query": {
                "type": "string",
                "description": "A natural language question or keyword search.",
            },
        },
        "required": ["document_id", "query"],
    }

    def run(self, document_id: str, query: str) -> str:  # noqa: ARG002
        raise NotImplementedError(
            "FilterByDocument.run: implement by applying metadata_filter "
            "{'document_id': document_id} to DocumentRepository.hybrid_search() "
            "or get_all(), then return formatted results."
        )


class FilterByAnnex:
    """Search within a specific annex only.

    Useful when the question is clearly about a particular annex
    (e.g. Annex I for climate change mitigation activities).
    """

    name = "filter_by_annex"
    description = (
        "Search within a specific annex only. "
        "Use this to scope retrieval to a single annex "
        "when you know which annex contains the answer."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "annex": {
                "type": "string",
                "description": (
                    "The annex identifier, e.g. 'Annex I', 'Annex II', 'Annex III'."
                ),
            },
            "query": {
                "type": "string",
                "description": "A natural language question or keyword search.",
            },
        },
        "required": ["annex", "query"],
    }

    def run(self, annex: str, query: str) -> str:  # noqa: ARG002
        raise NotImplementedError(
            "FilterByAnnex.run: implement by applying metadata_filter "
            "{'annex': annex} to DocumentRepository.hybrid_search() "
            "or get_all(), then return formatted results."
        )
