from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    question_id: str
    question: str
    answer: str
    sources_retrieved: list[dict]
    key_facts_found: list[str]
    key_facts_missing: list[str]
    recall_score: float          # len(found) / len(all key_facts)
    notes: str


class EvaluationHarness:
    """Runs the golden dataset against a RAG pipeline and reports results.

    The key_facts check is intentionally simple: substring presence in the answer.
    The goal is to make human evaluation easy, not to replace it.
    """

    def __init__(self, dataset_path: str, rag_pipeline) -> None:
        self.dataset_path = dataset_path
        self.rag_pipeline = rag_pipeline
        with open(dataset_path, encoding="utf-8") as f:
            self.dataset = json.load(f)

    def run(self) -> list[EvalResult]:
        """Run each golden dataset question through the RAG pipeline."""
        results: list[EvalResult] = []
        for item in self.dataset:
            response = self.rag_pipeline.query(item["question"])
            answer = response.get("answer", "")
            sources = response.get("sources", [])

            found = [
                fact for fact in item["key_facts"]
                if fact.lower() in answer.lower()
            ]
            missing = [
                fact for fact in item["key_facts"]
                if fact.lower() not in answer.lower()
            ]
            recall = len(found) / len(item["key_facts"]) if item["key_facts"] else 0.0

            results.append(EvalResult(
                question_id=item["id"],
                question=item["question"],
                answer=answer,
                sources_retrieved=sources,
                key_facts_found=found,
                key_facts_missing=missing,
                recall_score=recall,
                notes=item.get("notes", ""),
            ))
        return results

    def report(self, results: list[EvalResult]) -> None:
        """Print a readable comparison table to stdout."""
        col_w = 60
        sep = "-" * (col_w + 30)
        print(sep)
        print(f"{'EVALUATION REPORT':^{col_w + 30}}")
        print(sep)
        for r in results:
            print(f"\n[{r.question_id}] {r.question[:col_w]}")
            print(f"  Recall:  {r.recall_score:.0%} ({len(r.key_facts_found)}/{len(r.key_facts_found) + len(r.key_facts_missing)} key facts found)")
            print(f"  Sources: {len(r.sources_retrieved)} retrieved")
            if r.key_facts_missing:
                print("  Missing facts:")
                for fact in r.key_facts_missing:
                    print(f"    - {fact}")
            print(f"  Note: {r.notes}")
        print(sep)
        avg = sum(r.recall_score for r in results) / len(results) if results else 0.0
        print(f"  Average recall: {avg:.0%} across {len(results)} questions")
        print(sep)
