from __future__ import annotations

import psycopg
import psycopg_pool
from pgvector.psycopg import register_vector

from taxonomy_rag.config import settings

_pool: psycopg_pool.ConnectionPool | None = None


def _configure_conn(conn: psycopg.Connection) -> None:
    """Called for every new connection in the pool.

    Registers the pgvector type adapter so that reading an embedding column
    returns a list[float] instead of raw bytes.
    """
    register_vector(conn)


def get_pool() -> psycopg_pool.ConnectionPool:
    """Return the process-wide connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = psycopg_pool.ConnectionPool(
            conninfo=settings.dsn,
            min_size=1,
            max_size=10,
            configure=_configure_conn,
        )
    return _pool


def close_pool() -> None:
    """Gracefully close all connections in the pool."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
