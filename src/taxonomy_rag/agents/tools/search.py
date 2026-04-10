"""LangChain tools for searching the EU Taxonomy corpus.

STUB — implement these tools to enable agent-based retrieval.

Suggested approach:
    Use @tool from langchain_core.tools to define each function.
    Each tool wraps a DocumentRepository method and returns a formatted string
    or list that the agent can reason over.
"""

from langchain_core.tools import tool


@tool
def search_taxonomy(query: str) -> str:
    """Search the EU Taxonomy corpus for documents relevant to the query.

    Args:
        query: A natural language question or keyword search.

    Returns:
        A formatted string of the most relevant document excerpts.
    """
    raise NotImplementedError(
        "search_taxonomy: implement this tool to perform hybrid search "
        "against the documents table and return formatted results."
    )


@tool
def get_article(article_reference: str) -> str:
    """Retrieve the text of a specific article or annex section.

    Args:
        article_reference: e.g. 'Article 18', 'Annex I Activity 4.3', 'Article 3(1)'

    Returns:
        The text of the referenced article or section, or a not-found message.
    """
    raise NotImplementedError(
        "get_article: implement this tool to filter documents by metadata "
        "fields (article, annex) and return the matching chunk content."
    )
