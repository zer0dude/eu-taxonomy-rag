"""EU Taxonomy compliance checker agent.

STUB — implement this class to enable agent-based compliance analysis.

Suggested approach:
    Use a LangChain ReAct or tool-calling agent with the tools from
    taxonomy_rag.agents.tools. The agent should be able to:
    1. Identify the relevant economic activity in the EU Taxonomy
    2. Retrieve the applicable technical screening criteria
    3. Check DNSH criteria for all six environmental objectives
    4. Verify minimum safeguards (Article 18, Regulation 2020/852)
    5. Return a structured compliance assessment
"""

from taxonomy_rag.agents.tools.filters import filter_by_annex, filter_by_document
from taxonomy_rag.agents.tools.search import get_article, search_taxonomy

TOOLS = [search_taxonomy, get_article, filter_by_document, filter_by_annex]


class ComplianceCheckerAgent:
    """LangChain agent that assesses EU Taxonomy alignment for economic activities.

    Uses the search and filter tools to navigate the corpus and produce a
    structured compliance report covering:
    - Substantial contribution criteria
    - Do No Significant Harm (DNSH) for all six objectives
    - Minimum safeguards (Article 18)
    """

    def __init__(self, llm_provider: str | None = None) -> None:
        self.llm_provider = llm_provider

    def check(self, activity_description: str) -> dict:
        """Run a compliance check for the described economic activity.

        Args:
            activity_description: Natural language description of the activity
                                   to assess (e.g. "offshore wind farm, 500 MW").

        Returns:
            A dict with keys: activity, substantial_contribution, dnsh, safeguards,
            overall_alignment, sources.
        """
        raise NotImplementedError(
            "ComplianceCheckerAgent.check: implement the LangChain agent loop here. "
            "Wire up TOOLS with get_llm(self.llm_provider) and invoke the agent "
            "with activity_description as the input. "
            "See module docstring for the expected output structure."
        )
