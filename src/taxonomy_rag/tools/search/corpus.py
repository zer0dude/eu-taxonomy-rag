from __future__ import annotations

from taxonomy_rag.retrieval.base import Retriever
from taxonomy_rag.retrieval.scope import CorpusScope
from taxonomy_rag.tracing.base import NullTracer, Tracer

_DEFAULT_NAME = "search_corpus"
_DEFAULT_DESCRIPTION = (
    "Search the ingested EU Taxonomy document corpus for chunks relevant to "
    "a query. Returns numbered excerpts with document identifier, page range, "
    "and relevance score. Use this to retrieve official regulatory text before "
    "making any compliance claim."
)


class SearchCorpusTool:
    """Agent tool: search over a scoped EU Taxonomy document corpus.

    name and description are constructor parameters so that multiple instances
    with different retrieval strategies can coexist in one ToolKit without name
    collision. The retrieval strategy and corpus scope are also injected, making
    this class a pure composition point.

    Rebuild per answer() call (passing the current tracer) so that token usage
    from retrieval-internal LLM calls (e.g. HyDE) is attributed to the run.
    """

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
        name: str = _DEFAULT_NAME,
        description: str = _DEFAULT_DESCRIPTION,
        tracer: Tracer = NullTracer(),
    ) -> None:
        self.name = name
        self.description = description
        self._retrieval = retrieval
        self._scope = scope
        self._tracer = tracer

    def run(self, query: str) -> str:
        results = self._retrieval.retrieve(query, self._scope, self._tracer)
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
