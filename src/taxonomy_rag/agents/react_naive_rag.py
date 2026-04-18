"""ReAct agent with naive vector corpus search.

Agent composition:
  - Core loop:       ReAct (AgentLoop / litellm tool-use)
  - DB retrieval:    SearchCorpusTool(NaiveVectorRetrieval, NAIVE_PDF_CORPUS)
  - Attachment tool: ReadFullDocument (added per-question when attachments present)
  - Default prompt:  prompts/compliance_v1.txt

Naming convention: react_naive_rag
  react        = AgentLoop ReAct core
  naive_rag    = NaiveVectorRetrieval scoped to naive_pdf corpus
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from taxonomy_rag.llm.loop import AgentLoop
from taxonomy_rag.llm.provider import get_completion_kwargs
from taxonomy_rag.readers.base import AttachmentInfo
from taxonomy_rag.readers.registry import default_registry
from taxonomy_rag.retrieval.naive import NaiveVectorRetrieval
from taxonomy_rag.retrieval.scope import NAIVE_PDF_CORPUS
from taxonomy_rag.tools.attachment.read_full import ReadFullDocument
from taxonomy_rag.tools.base import ToolKit
from taxonomy_rag.tools.search.corpus import SearchCorpusTool
from taxonomy_rag.tracing.base import NullTracer

_PROMPT_PATH = Path(__file__).parent.parent.parent.parent / "prompts" / "compliance_v1.txt"


def get_agent() -> "ReactNaiveRagAgent":
    load_dotenv()
    return ReactNaiveRagAgent()


class ReactNaiveRagAgent:
    """ReAct agent that searches the ingested EU Taxonomy corpus as its primary tool.

    For questions with attachments (e.g. hard_01), ReadFullDocument is added to
    the toolkit automatically so the agent can read supporting evidence alongside
    the corpus search results.

    The default system prompt (compliance_v1.txt) requires the agent to cite
    every claim from retrieved text. Pass ``prompt`` to answer() to override
    for experimental runs.
    """

    _DEFAULT_PROMPT: str = (
        _PROMPT_PATH.read_text(encoding="utf-8") if _PROMPT_PATH.exists() else ""
    )

    def __init__(self) -> None:
        self._loop = AgentLoop(completion_kwargs=get_completion_kwargs())
        self.registry = default_registry()
        retrieval = NaiveVectorRetrieval(top_k=5)
        self._search_tool = SearchCorpusTool(retrieval, NAIVE_PDF_CORPUS)

    def answer(
        self,
        question: str,
        context: str = "",
        prompt: str = "",
        attachments: list[AttachmentInfo] = [],
        tracer: NullTracer = NullTracer(),
    ) -> str:
        system_prompt = prompt or self._DEFAULT_PROMPT

        tools: list = [self._search_tool]
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
