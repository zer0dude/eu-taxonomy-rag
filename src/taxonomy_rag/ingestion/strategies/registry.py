"""Strategy registry — the single place to register ingestion strategies.

To add a new strategy:
    1. Create a new file in this directory (e.g. structural_pdf.py)
    2. Add one line to DEFAULT_REGISTRY below

Scripts and tests import DEFAULT_REGISTRY directly; they never change when new
strategies are added.
"""

from __future__ import annotations

from taxonomy_rag.ingestion.strategies.base import IngestionStrategy
from taxonomy_rag.ingestion.strategies.naive_pdf import NaivePDFStrategy


class StrategyRegistry:
    """Maps strategy names to IngestionStrategy instances and auto-dispatches
    by file type via ``find_for_file``."""

    def __init__(self, strategies: list[IngestionStrategy]) -> None:
        self._strategies: dict[str, IngestionStrategy] = {
            s.name: s for s in strategies
        }

    def get(self, name: str) -> IngestionStrategy:
        """Return the strategy for the given name, or raise ValueError."""
        if name not in self._strategies:
            available = ", ".join(self._strategies)
            raise ValueError(
                f"Unknown strategy: {name!r}. Available: {available}"
            )
        return self._strategies[name]

    def find_for_file(self, file_path: str) -> IngestionStrategy | None:
        """Return the first strategy that supports this file, or None."""
        for strategy in self._strategies.values():
            if strategy.supports(file_path):
                return strategy
        return None

    def names(self) -> list[str]:
        """Return all registered strategy names."""
        return list(self._strategies)


# ---------------------------------------------------------------------------
# The canonical registry — add new strategies here
# ---------------------------------------------------------------------------
DEFAULT_REGISTRY = StrategyRegistry([
    NaivePDFStrategy(),
    # StructuralPDFStrategy(),    # uncomment when implemented
    # HierarchicalPDFStrategy(),  # uncomment when implemented
])
