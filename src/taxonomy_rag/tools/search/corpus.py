from __future__ import annotations

from taxonomy_rag.retrieval.base import Retriever
from taxonomy_rag.retrieval.scope import CorpusScope


class SearchCorpusTool:
    """Agent tool: semantic search over the ingested EU Taxonomy document corpus.

    Composes any Retriever implementation with a CorpusScope that restricts
    which chunks are visible to this tool instance. Returns a formatted string
    of numbered results, each with document ID, page range, similarity score,
    and chunk text.

    Swap the retrieval argument to change the search strategy without touching
    the agent or the tool interface.
    """

    name = "search_corpus"
    description = (
        "Search the ingested EU Taxonomy document corpus for chunks relevant to "
        "a query. Returns numbered excerpts with document identifier, page range, "
        "and relevance score. Use this to retrieve official regulatory text before "
        "making any compliance claim."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A natural-language search query. Phrase it as a question or "
                    "keyword string describing the regulatory text you need."
                ),
            }
        },
        "required": ["query"],
    }

    def __init__(
        self,
        retrieval: Retriever,
        scope: CorpusScope,
    ) -> None:
        self._retrieval = retrieval
        self._scope = scope

    def run(self, query: str) -> str:
        results = self._retrieval.retrieve(query, self._scope)
        if not results:
            return "No relevant documents found."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            doc_id = r.metadata.get("document_id", "unknown")
            page_range = r.metadata.get("page_range", "")
            header = f"[{i}] Document: {doc_id}"
            if page_range:
                header += f" | Pages: {page_range}"
            header += f" | Score: {r.score:.3f}"
            lines.append(header)
            lines.append(r.content)
            lines.append("")

        return "\n".join(lines).rstrip()
