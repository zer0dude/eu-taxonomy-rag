CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id               BIGSERIAL    PRIMARY KEY,
    content          TEXT         NOT NULL,
    metadata         JSONB        NOT NULL DEFAULT '{}',
    embedding        vector(384)  NOT NULL,
    tsvector_content tsvector     GENERATED ALWAYS AS (
                         to_tsvector('english', content)
                     ) STORED,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- HNSW index for approximate nearest-neighbour search (cosine distance)
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS documents_tsvector_gin_idx
    ON documents USING gin (tsvector_content);

-- GIN index for JSONB metadata filtering
CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx
    ON documents USING gin (metadata);
