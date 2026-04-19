"""ReAct agent with hybrid BM25+vector search over the naively-chunked corpus.

Agent composition:
  - Core loop:       ReAct (AgentLoop / litellm tool-use)
  - DB retrieval:    SearchCorpusTool(HybridRetrieval, NAIVE_CORPUS)
                     tool name: search_naive_corpus_hybrid
  - Attachment tool: ReadFullDocument (added per-question when attachments present)
  - Default prompt:  prompts/compliance_v1.txt

Naming convention: react_{ingestion}_{corpus}_{retrieval}_rag
  naive_corpus = naively-chunked naive_pdf ingestion strategy
  hybrid       = BM25 full-text + cosine vector, fused with RRF (HybridRetrieval)
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from taxonomy_rag.llm.loop import AgentLoop
from taxonomy_rag.llm.provider import get_completion_kwargs
from taxonomy_rag.readers.base import AttachmentInfo
from taxonomy_rag.readers.registry import default_registry
from taxonomy_rag.retrieval.hybrid import HybridRetrieval
from taxonomy_rag.retrieval.scope import NAIVE_CORPUS
from taxonomy_rag.tools.attachment.read_full import ReadFullDocument
from taxonomy_rag.tools.base import ToolKit
from taxonomy_rag.tools.search.corpus import SearchCorpusTool
from taxonomy_rag.tracing.base import NullTracer, Tracer

_PROMPT_PATH = Path(__file__).parents[3] / "prompts" / "compliance_v1.txt"

_TOOL_NAME = "search_naive_corpus_hybrid"
_TOOL_DESCRIPTION = (
    "Hybrid BM25 + vector search (Reciprocal Rank Fusion) over the naively-chunked "
    "EU Taxonomy corpus (naive_pdf ingestion). Combines keyword and semantic ranking — "
    "better than pure vector search for queries containing specific regulation codes, "
    "article numbers, or exact terminology."
)


def get_agent() -> "ReactNaiveCorpusHybridRagAgent":
    load_dotenv()
    return ReactNaiveCorpusHybridRagAgent()


class ReactNaiveCorpusHybridRagAgent:
    """ReAct agent: hybrid BM25+vector RRF search over the naively-chunked EU Taxonomy corpus.

    For questions with attachments, ReadFullDocument is added to the toolkit
    automatically. The search tool is rebuilt per answer() call so the tracer
    is correctly injected for token accounting.
    """

    _DEFAULT_PROMPT: str = (
        _PROMPT_PATH.read_text(encoding="utf-8") if _PROMPT_PATH.exists() else ""
    )

    def __init__(self) -> None:
        self._loop = AgentLoop(completion_kwargs=get_completion_kwargs())
        self.registry = default_registry()
        self._retrieval = HybridRetrieval(top_k=5)

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

        search_tool = SearchCorpusTool(
            self._retrieval,
            NAIVE_CORPUS,
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            tracer=tracer,
        )
        tools: list = [search_tool]
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
