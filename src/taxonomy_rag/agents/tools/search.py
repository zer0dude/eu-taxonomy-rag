"""Corpus search tools for agent-based retrieval.

STUB — implement run() in each class to enable retrieval.

These tools conform to the Tool protocol in tools/base.py and are registered
with ToolKit for use by any AgentLoop-based agent.
"""

from __future__ import annotations


class SearchTaxonomy:
    """Search the EU Taxonomy corpus for documents relevant to a query.

    Returns a formatted string of the most relevant document excerpts,
    drawn from all ingested source documents.
    """

    name = "search_taxonomy"
    description = (
        "Search the EU Taxonomy corpus for documents relevant to the query. "
        "Returns a formatted string of the most relevant document excerpts "
        "with source references. Use this for broad semantic search across "
        "all ingested documents."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A natural language question or keyword search.",
            }
        },
        "required": ["query"],
    }

    def run(self, query: str) -> str:  # noqa: ARG002
        raise NotImplementedError(
            "SearchTaxonomy.run: implement hybrid search against the documents "
            "table (DocumentRepository.hybrid_search) and return formatted results. "
            "Each result should include document_id, article/annex reference if "
            "available in metadata, and the chunk text."
        )


class GetArticle:
    """Retrieve the text of a specific article or annex section by reference.

    Use this when the agent already knows the specific article it needs,
    e.g. after a search revealed the relevant article number.
    """

    name = "get_article"
    description = (
        "Retrieve the text of a specific article or annex section. "
        "e.g. 'Article 10', 'Annex I Activity 4.3', 'Article 3(1)'. "
        "Use this when you know the exact article reference you need."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "article_reference": {
                "type": "string",
                "description": (
                    "The article or annex reference to retrieve. "
                    "Examples: 'Article 10', 'Annex I', 'Article 3(1)'."
                ),
            }
        },
        "required": ["article_reference"],
    }

    def run(self, article_reference: str) -> str:  # noqa: ARG002
        raise NotImplementedError(
            "GetArticle.run: implement metadata-filtered lookup using "
            "DocumentRepository.get_all(metadata_filter={'article': ...}) "
            "or similar, and return the matching chunk content."
        )
