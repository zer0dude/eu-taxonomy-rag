"""Integration test configuration — skips all tests if PostgreSQL is unreachable."""

from __future__ import annotations

import pytest


def _is_db_available() -> bool:
    try:
        import psycopg
        from taxonomy_rag.config import settings
        with psycopg.connect(settings.dsn, connect_timeout=2):
            return True
    except Exception:
        return False


DB_AVAILABLE = _is_db_available()

skip_if_no_db = pytest.mark.skipif(
    not DB_AVAILABLE,
    reason="PostgreSQL not reachable — skipping integration tests. Run: docker compose up -d",
)
