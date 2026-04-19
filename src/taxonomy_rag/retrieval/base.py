from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from taxonomy_rag.retrieval.scope import CorpusScope


@dataclass
class RetrievalResult:
    doc_id: int
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    """Interface that all retrieval implementations must satisfy.

    A Retriever embeds a query, searches the document store, and returns
    ranked results. It has no opinion on how results are used — that is
    the tool's concern.

    Implement this protocol in retrieval/ to add a new retrieval method
    (e.g. HybridRetrieval, AdvancedRetrieval). Wrap with SearchCorpusTool
    to expose as an agent tool.
    """

    def retrieve(
        self,
        query: str,
        scope: "CorpusScope | None" = None,
    ) -> list[RetrievalResult]:
        """Search for documents relevant to query, optionally scoped to a corpus subset."""
        ...
