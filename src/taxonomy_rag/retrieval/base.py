from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from taxonomy_rag.retrieval.scope import CorpusScope


@dataclass
class RetrievalResult:
    doc_id: int
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
