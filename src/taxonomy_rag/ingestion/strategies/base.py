from typing import Protocol

from taxonomy_rag.ingestion.pipeline import IngestionPipeline


class IngestionStrategy(Protocol):
    """A named, self-contained ingestion recipe.

    An IngestionStrategy is a composition of a specific parser and a specific
    chunker. Its ``name`` is stored in every chunk's metadata as
    ``ingestion_strategy``, making it the primary key for filtering and
    comparing results across experiments.

    Add new strategies as new files in this directory — never modify existing
    ones. Register them in registry.py's DEFAULT_REGISTRY.
    """

    name: str         # stored in DB: metadata["ingestion_strategy"]
    description: str  # shown by --list-strategies

    def supports(self, file_path: str) -> bool:
        """Return True if this strategy can ingest the given file."""
        ...

    def build_pipeline(self) -> IngestionPipeline:
        """Return a fully configured IngestionPipeline for this strategy."""
        ...
