"""LangChain tools for filtering the EU Taxonomy corpus by metadata.

STUB — implement these tools to enable scoped agent retrieval.

These tools let the agent narrow its search to a specific source document
or annex before running a semantic search, reducing noise in retrieval.
"""

from langchain_core.tools import tool


@tool
def filter_by_document(document_id: str, query: str) -> str:
    """Search within a specific source document only.

    Args:
        document_id: The document identifier, e.g. '2021_2139' or '2020_852'.
        query:       A natural language question or keyword search.

    Returns:
        A formatted string of relevant excerpts from that document only.
    """
    raise NotImplementedError(
        "filter_by_document: implement this tool to apply a metadata_filter "
        "{'document_id': document_id} to DocumentRepository.get_all() or "
        "hybrid_search(), then return formatted results."
    )


@tool
def filter_by_annex(annex: str, query: str) -> str:
    """Search within a specific annex only.

    Args:
        annex: The annex identifier, e.g. 'Annex I', 'Annex II', 'Annex III'.
        query: A natural language question or keyword search.

    Returns:
        A formatted string of relevant excerpts from that annex only.
    """
    raise NotImplementedError(
        "filter_by_annex: implement this tool to apply a metadata_filter "
        "{'annex': annex} to the search and return formatted results."
    )
