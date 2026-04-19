"""ReAct agent with both hybrid and advanced search over the naively-chunked corpus.

Agent composition:
  - Core loop:       ReAct (AgentLoop / litellm tool-use)
  - DB retrieval 1:  SearchCorpusTool(HybridRetrieval, NAIVE_CORPUS)
                     tool name: search_naive_corpus_hybrid
  - DB retrieval 2:  SearchCorpusTool(AdvancedRetrieval, NAIVE_CORPUS)
                     tool name: search_naive_corpus_advanced
  - Attachment tool: ReadFullDocument (added per-question when attachments present)
  - Default prompt:  prompts/compliance_v1.txt

Naming convention: react_{ingestion}_{corpus}_{retrieval}_rag
  naive_corpus = naively-chunked naive_pdf ingestion strategy
  multi        = agent chooses between hybrid and advanced per query

The agent selects the retrieval tool it judges most appropriate for each
query — hybrid for keyword-rich searches (regulation codes, article numbers),
advanced when semantic precision matters most.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from taxonomy_rag.llm.loop import AgentLoop
from taxonomy_rag.llm.provider import get_completion_kwargs
from taxonomy_rag.readers.base import AttachmentInfo
from taxonomy_rag.readers.registry import default_registry
from taxonomy_rag.retrieval.advanced import AdvancedRetrieval
from taxonomy_rag.retrieval.hybrid import HybridRetrieval
from taxonomy_rag.retrieval.scope import NAIVE_CORPUS
from taxonomy_rag.tools.attachment.read_full import ReadFullDocument
from taxonomy_rag.tools.base import ToolKit
from taxonomy_rag.tools.search.corpus import SearchCorpusTool
from taxonomy_rag.tracing.base import NullTracer, Tracer

_PROMPT_PATH = Path(__file__).parents[3] / "prompts" / "compliance_v1.txt"

_HYBRID_NAME = "search_naive_corpus_hybrid"
_HYBRID_DESCRIPTION = (
    "Hybrid BM25 + vector search (Reciprocal Rank Fusion) over the naively-chunked "
    "EU Taxonomy corpus (naive_pdf ingestion). Combines keyword and semantic ranking — "
    "best for queries containing specific regulation codes, article numbers, or exact terminology."
)

_ADVANCED_NAME = "search_naive_corpus_advanced"
_ADVANCED_DESCRIPTION = (
    "Advanced search with HyDE query expansion and cross-encoder reranking over the "
    "naively-chunked EU Taxonomy corpus (naive_pdf ingestion). HyDE generates a "
    "hypothetical answer paragraph to improve recall; cross-encoder reranking selects "
    "the most relevant passages from candidates. Highest precision, higher latency — "
    "best for conceptual or criteria-based questions."
)


def get_agent() -> "ReactNaiveCorpusMultiRagAgent":
    load_dotenv()
    return ReactNaiveCorpusMultiRagAgent()


class ReactNaiveCorpusMultiRagAgent:
    """ReAct agent with hybrid and advanced search tools over the naively-chunked corpus.

    The agent selects the appropriate retrieval tool per query. Hybrid is faster
    and keyword-aware; advanced has higher semantic precision but calls the LLM
    internally for HyDE and runs cross-encoder reranking.

    Both retrieval instances are created once in __init__; tools are rebuilt per
    answer() call to inject the current tracer.
    """

    _DEFAULT_PROMPT: str = (
        _PROMPT_PATH.read_text(encoding="utf-8") if _PROMPT_PATH.exists() else ""
    )

    def __init__(self) -> None:
        self._loop = AgentLoop(completion_kwargs=get_completion_kwargs())
        self.registry = default_registry()
        self._hybrid_retrieval = HybridRetrieval(top_k=5)
        self._advanced_retrieval = AdvancedRetrieval(top_k=5)

    def answer(
        self,
        question: str,
        context: str = "",
        prompt: str = "",
        attachments: list[AttachmentInfo] | None = None,
        tracer: Tracer = NullTracer(),
    ) -> str:
        attachments = attachments or []
        system_prompt = prompt or self._DEFAULT_PROMPT

        hybrid_tool = SearchCorpusTool(
            self._hybrid_retrieval,
            NAIVE_CORPUS,
            name=_HYBRID_NAME,
            description=_HYBRID_DESCRIPTION,
            tracer=tracer,
        )
        advanced_tool = SearchCorpusTool(
            self._advanced_retrieval,
            NAIVE_CORPUS,
            name=_ADVANCED_NAME,
            description=_ADVANCED_DESCRIPTION,
            tracer=tracer,
        )
        tools: list = [hybrid_tool, advanced_tool]
        if attachments:
            path_map = {a.name: a.path for a in attachments}
            tools.append(ReadFullDocument(path_map, self.registry))

        toolkit = ToolKit(tools)

        parts: list[str] = []
        if context:
            parts.append(f"Context:\n{context}")
        if attachments:
            att_lines = [
                f"  - {a.name} ({a.file_type}, {a.size_bytes // 1024} KB)"
                for a in attachments
            ]
            parts.append("Available attachments:\n" + "\n".join(att_lines))
        parts.append(f"Question:\n{question}")
        user_content = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        return self._loop.run(messages=messages, toolkit=toolkit, tracer=tracer)
