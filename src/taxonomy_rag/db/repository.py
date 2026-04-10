from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from taxonomy_rag.db.connection import get_pool


class DocumentRepository:
    """CRUD and vector/hybrid search operations on the documents table."""

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def insert(
        self,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert a document and return its generated id."""
        with get_pool().connection() as conn:
            row = conn.execute(
                """
                INSERT INTO documents (content, embedding, metadata)
                VALUES (%s, %s::vector, %s)
                RETURNING id
                """,
                (content, embedding, Jsonb(metadata or {})),
            ).fetchone()
        return row[0]

    def delete(self, doc_id: int) -> bool:
        """Delete a document by id. Returns True if a row was deleted."""
        with get_pool().connection() as conn:
            result = conn.execute(
                "DELETE FROM documents WHERE id = %s", (doc_id,)
            )
        return result.rowcount > 0

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_by_id(self, doc_id: int) -> dict | None:
        with get_pool().connection() as conn:
            conn.row_factory = dict_row
            return conn.execute(
                "SELECT id, content, metadata, created_at FROM documents WHERE id = %s",
                (doc_id,),
            ).fetchone()

    def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Fetch documents with optional JSONB metadata filter.

        metadata_filter example: {"document_id": "2021_2139"}
        Uses the @> (contains) operator against the metadata GIN index.
        """
        with get_pool().connection() as conn:
            conn.row_factory = dict_row
            if metadata_filter:
                return conn.execute(
                    """
                    SELECT id, content, metadata, created_at
                    FROM documents
                    WHERE metadata @> %s::jsonb
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (Jsonb(metadata_filter), limit, offset),
                ).fetchall()
            return conn.execute(
                """
                SELECT id, content, metadata, created_at
                FROM documents
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            ).fetchall()

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def vector_search(
        self, embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        """Cosine similarity search.

        pgvector's <=> operator is cosine *distance* (0 = identical, 2 = opposite).
        We convert to similarity: score = 1 - distance, so 1.0 is a perfect match.
        """
        with get_pool().connection() as conn:
            conn.row_factory = dict_row
            return conn.execute(
                """
                SELECT id, content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            ).fetchall()

    # ------------------------------------------------------------------
    # Hybrid search (BM25 full-text + cosine vector, fused with RRF)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_text: str,
        embedding: list[float],
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of BM25 full-text and cosine vector search.

        RRF formula: score(d) = Σ 1 / (k + rank_i(d))
        rrf_k=60 is the standard constant from Cormack et al. 2009.

        Pre-fetches top_k * 4 candidates from each ranker before fusion so that
        documents appearing in only one list still get a fair chance.
        """
        pre_k = top_k * 4
        with get_pool().connection() as conn:
            conn.row_factory = dict_row
            return conn.execute(
                """
                WITH
                vector_ranked AS (
                    SELECT id, content, metadata,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> %(emb)s::vector) AS rank
                    FROM documents
                    ORDER BY embedding <=> %(emb)s::vector
                    LIMIT %(pre_k)s
                ),
                fts_ranked AS (
                    SELECT id, content, metadata,
                           ROW_NUMBER() OVER (
                               ORDER BY ts_rank_cd(tsvector_content,
                                        plainto_tsquery('english', %(query)s)) DESC
                           ) AS rank
                    FROM documents
                    WHERE tsvector_content @@ plainto_tsquery('english', %(query)s)
                    ORDER BY ts_rank_cd(tsvector_content,
                             plainto_tsquery('english', %(query)s)) DESC
                    LIMIT %(pre_k)s
                ),
                fused AS (
                    SELECT
                        COALESCE(v.id,      f.id)        AS id,
                        COALESCE(v.content, f.content)   AS content,
                        COALESCE(v.metadata, f.metadata) AS metadata,
                        COALESCE(1.0 / (%(k)s + v.rank), 0) +
                        COALESCE(1.0 / (%(k)s + f.rank), 0) AS rrf_score
                    FROM vector_ranked v
                    FULL OUTER JOIN fts_ranked f ON v.id = f.id
                )
                SELECT id, content, metadata, rrf_score AS score
                FROM fused
                ORDER BY rrf_score DESC
                LIMIT %(top_k)s
                """,
                {"emb": embedding, "query": query_text, "pre_k": pre_k,
                 "k": rrf_k, "top_k": top_k},
            ).fetchall()
