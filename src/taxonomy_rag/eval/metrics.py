"""Advanced evaluation metrics.

STUB — implement these functions to enable quantitative RAG evaluation.

These complement the simple substring recall in EvaluationHarness with
proper IR metrics and LLM-as-judge scoring.
"""

from __future__ import annotations


def recall_at_k(relevant_ids: list[int], retrieved_ids: list[int], k: int) -> float:
    """Fraction of relevant documents found in the top-k retrieved results.

    Args:
        relevant_ids:  Ground-truth document ids that should be retrieved.
        retrieved_ids: Ordered list of retrieved document ids (most relevant first).
        k:             Cut-off rank.

    Returns:
        Recall@k in [0, 1].
    """
    raise NotImplementedError(
        "recall_at_k: implement as len(set(relevant_ids) & set(retrieved_ids[:k])) / len(relevant_ids)."
    )


def mrr(relevant_ids: list[int], retrieved_ids: list[int]) -> float:
    """Mean Reciprocal Rank for a single query.

    Args:
        relevant_ids:  Ground-truth document ids.
        retrieved_ids: Ordered list of retrieved document ids.

    Returns:
        1/rank of the first relevant document, or 0 if none found.
    """
    raise NotImplementedError(
        "mrr: implement by finding the first position in retrieved_ids that is in "
        "relevant_ids and returning 1 / (position + 1)."
    )


def llm_judge(question: str, answer: str, reference_facts: list[str], llm_provider: str | None = None) -> dict:
    """Use an LLM to score the answer against reference facts.

    Args:
        question:        The original question.
        answer:          The RAG-generated answer to evaluate.
        reference_facts: Key facts the answer should cover.
        llm_provider:    LLM provider override (defaults to settings.llm_provider).

    Returns:
        {"score": float, "reasoning": str} where score is in [0, 1].
    """
    raise NotImplementedError(
        "llm_judge: implement by prompting the LLM to score how well `answer` covers "
        "`reference_facts` relative to `question`. Return a structured score + reasoning."
    )
