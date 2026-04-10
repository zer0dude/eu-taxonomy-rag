# EU Taxonomy RAG — Project Brief

## Purpose of this document

This document is a complete brief for scaffolding a new Python project from scratch.
It describes the project goal, all technical decisions already made, the full directory
structure to create, and exactly what should be implemented vs left as a stub.

Read it fully before writing any code.

---

## Project goal

A research and experimentation platform for applying **Retrieval-Augmented Generation
(RAG)** and **LangChain agents** to the **EU Taxonomy for Sustainable Finance**.

The EU Taxonomy is a classification system defined in EU Regulation 2020/852 and its
associated Delegated Acts (primarily 2021/2139 — the Climate Delegated Act). It defines
which economic activities qualify as environmentally sustainable, specifying technical
screening criteria, Do No Significant Harm (DNSH) criteria, and minimum safeguards.

The corpus consists of long, complex regulatory PDFs with:
- Recitals, articles, annexes, sub-annexes
- Multi-column tables of screening criteria (one of the hardest parts to parse)
- Cross-references between documents ("as referred to in Article 3 of Regulation...")
- Corrigenda (official corrections that amend earlier documents)

The owner of this project is a solo developer learning about Postgres, pgvector, RAG
pipelines, and LangChain agents. The project is explicitly experimental — different
ingestion strategies, retrieval approaches, and agent designs will be swapped in and
out to compare their behaviour on the same domain.

---

## What this project is NOT

- Not a production system. No auth, no multi-tenancy, no deployment pipeline.
- Not a fixed implementation. The ingestion and agent layers are deliberately left as
  stubs with clear interfaces — the developer will fill them with different strategies.
- Not a re-implementation of the learning playground. This project builds on those
  lessons but starts fresh.

---

## Technical decisions (already made, do not change)

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Package manager | `uv` |
| Database | PostgreSQL 16 + pgvector extension |
| DB driver | `psycopg` (v3) + `psycopg_pool` |
| Vector type adapter | `pgvector` (`pgvector.psycopg`) |
| ORM / query style | Raw SQL via psycopg — no ORM |
| Embeddings | `sentence-transformers`, default model `all-MiniLM-L6-v2` (384 dims) |
| LLM abstraction | LangChain (`langchain-core`) with swappable providers |
| LLM providers | Ollama (local), Anthropic, OpenAI — selected via `.env` |
| Config | `pydantic-settings`, reads from `.env` |
| API | FastAPI (optional thin layer, same pattern as playground) |
| Containerisation | Docker Compose — Postgres only; Python runs on the host |
| PDF parsing | `pymupdf` (fitz) as the baseline parser — others may be added |
| Spreadsheet parsing | `openpyxl` for `.xlsx` |
| Settings structure | Same pattern as reference playground (see Reference section) |

---

## Directory structure to create

Create every file and directory listed below. Files marked `# STUB` should exist with
the correct imports and class/function signatures but no real implementation — raise
`NotImplementedError` or return empty results. Files marked `# IMPLEMENT` should be
fully implemented. Files marked `# EMPTY` are data directories or placeholders.

```
eu-taxonomy-rag/
│
├── pyproject.toml                  # IMPLEMENT
├── .env.example                    # IMPLEMENT
├── .gitignore                      # IMPLEMENT
├── CLAUDE.md                       # IMPLEMENT  ← instructions for future AI sessions
│
├── docker/
│   └── postgres/
│       └── init/
│           └── 00_init.sql         # IMPLEMENT
│
├── docker-compose.yml              # IMPLEMENT
│
├── data/
│   ├── raw/                        # EMPTY — source PDFs and spreadsheets go here
│   │   └── .gitkeep
│   └── processed/                  # EMPTY — intermediate outputs go here
│       └── .gitkeep
│
├── eval/
│   └── golden_dataset.json         # IMPLEMENT — 3 example questions (see spec below)
│
├── scripts/
│   ├── ingest.py                   # STUB — CLI: python scripts/ingest.py <file>
│   └── evaluate.py                 # STUB — CLI: python scripts/evaluate.py
│
├── notebooks/
│   ├── 01_ingestion_experiments.ipynb   # IMPLEMENT — minimal, just setup cell + section headers
│   ├── 02_retrieval_experiments.ipynb   # IMPLEMENT — minimal
│   └── 03_agent_experiments.ipynb       # IMPLEMENT — minimal
│
└── src/
    └── taxonomy_rag/
        ├── __init__.py
        ├── config.py               # IMPLEMENT
        │
        ├── db/
        │   ├── __init__.py
        │   ├── connection.py       # IMPLEMENT — connection pool, same pattern as reference
        │   └── repository.py      # IMPLEMENT — DocumentRepository, same as reference
        │
        ├── embeddings/
        │   ├── __init__.py
        │   └── embedder.py         # IMPLEMENT — Embedder class, same as reference
        │
        ├── llm/
        │   ├── __init__.py
        │   └── provider.py         # IMPLEMENT — get_llm(), same as reference
        │
        ├── ingestion/
        │   ├── __init__.py
        │   ├── models.py           # IMPLEMENT — ParsedDocument, Chunk dataclasses
        │   ├── pipeline.py         # IMPLEMENT — IngestionPipeline orchestrator
        │   ├── parsers/
        │   │   ├── __init__.py
        │   │   ├── base.py         # IMPLEMENT — DocumentParser Protocol
        │   │   ├── pdf.py          # STUB — PDFParser(DocumentParser)
        │   │   └── spreadsheet.py  # STUB — SpreadsheetParser(DocumentParser)
        │   └── chunkers/
        │       ├── __init__.py
        │       ├── base.py         # IMPLEMENT — Chunker Protocol
        │       ├── naive.py        # STUB — NaiveChunker: fixed token size + overlap
        │       ├── structural.py   # STUB — StructuralChunker: split at article/annex boundaries
        │       └── hierarchical.py # STUB — HierarchicalChunker: small chunks + parent article
        │
        ├── rag/
        │   ├── __init__.py
        │   ├── naive.py            # IMPLEMENT — NaiveRAG, same as reference
        │   ├── hybrid.py           # IMPLEMENT — HybridRAG, same as reference
        │   └── advanced.py         # IMPLEMENT — AdvancedRAG (HyDE + reranking), same as reference
        │
        ├── agents/
        │   ├── __init__.py
        │   ├── tools/
        │   │   ├── __init__.py
        │   │   ├── search.py       # STUB — search_taxonomy(), get_article() LangChain tools
        │   │   └── filters.py      # STUB — filter_by_document(), filter_by_annex() tools
        │   └── compliance_checker.py  # STUB — ComplianceCheckerAgent
        │
        └── eval/
            ├── __init__.py
            ├── harness.py          # IMPLEMENT — EvaluationHarness (see spec below)
            └── metrics.py          # STUB — recall_at_k(), mrr(), llm_judge()
```

---

## Detailed specifications

### `pyproject.toml`

Use `uv` conventions. Package name `taxonomy-rag`, version `0.1.0`.

Required dependencies:
```
psycopg[binary]>=3.1
psycopg_pool>=3.1
pgvector>=0.3
pydantic-settings>=2.0
sentence-transformers>=3.0
langchain-core>=0.3
langchain-anthropic>=0.3
langchain-openai>=0.3
langchain-ollama>=0.3
langchain-community>=0.3
fastapi>=0.115
uvicorn>=0.30
pymupdf>=1.24
openpyxl>=3.1
python-dotenv>=1.0
```

Dev dependencies: `pytest`, `jupyter`, `ipykernel`

---

### `docker-compose.yml`

Single service: `postgres` using image `pgvector/pgvector:pg16`.
Environment variables read from `.env`.
Mount `./docker/postgres/init` to `/docker-entrypoint-initdb.d`.
Expose port 5432. Named volume for data persistence.

---

### `docker/postgres/init/00_init.sql`

```sql
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
```

---

### `config.py`

Extend the reference pattern with taxonomy-specific settings:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "taxonomy"
    db_user: str = "taxonomy"
    db_password: str = "taxonomy"

    @computed_field
    @property
    def dsn(self) -> str: ...

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384      # must match vector(N) in SQL

    # LLM
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Ingestion defaults — can be overridden per experiment
    default_chunk_size: int = 512       # tokens
    default_chunk_overlap: int = 50     # tokens
    default_chunker: str = "naive"      # "naive" | "structural" | "hierarchical"
```

---

### `ingestion/models.py`

```python
@dataclass
class ParsedDocument:
    """Raw output of a parser — one per source file."""
    source_path: str
    document_id: str        # e.g. "2021_2139" (regulation number, underscored)
    document_type: str      # "regulation" | "delegated_act" | "corrigendum" | "spreadsheet"
    title: str
    pages: list[str]        # raw text per page, in order
    metadata: dict          # any structured info extracted by the parser

@dataclass
class Chunk:
    """A single unit ready for embedding and storage."""
    content: str
    metadata: dict          # must include: source, document_id, chunk_strategy
                            # should include where known: article, annex, page_range
```

---

### `ingestion/parsers/base.py`

```python
from typing import Protocol
from taxonomy_rag.ingestion.models import ParsedDocument

class DocumentParser(Protocol):
    def parse(self, file_path: str) -> ParsedDocument: ...
    def supports(self, file_path: str) -> bool: ...
```

---

### `ingestion/chunkers/base.py`

```python
from typing import Protocol
from taxonomy_rag.ingestion.models import ParsedDocument, Chunk

class Chunker(Protocol):
    def chunk(self, document: ParsedDocument) -> list[Chunk]: ...
```

---

### `ingestion/pipeline.py`

The `IngestionPipeline` class orchestrates: parse → chunk → embed → store.

```python
class IngestionPipeline:
    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        embedder: Embedder,
        repo: DocumentRepository,
    ): ...

    def run(self, file_path: str) -> IngestionResult: ...
        # 1. parser.parse(file_path) → ParsedDocument
        # 2. chunker.chunk(document) → list[Chunk]
        # 3. embedder.embed_batch([c.content for c in chunks]) → list[list[float]]
        # 4. repo.insert(...) for each chunk
        # Returns IngestionResult(document_id, chunks_stored, errors)

@dataclass
class IngestionResult:
    document_id: str
    chunks_stored: int
    errors: list[str]
```

---

### `eval/golden_dataset.json`

Three realistic EU Taxonomy questions. The agent should be able to answer each
correctly if ingestion and retrieval work. The exact questions and format are still to be determined. This is what a json could look like:

```json
[
  {
    "id": "q1",
    "question": "Does offshore wind power generation qualify as a sustainable activity under the EU Taxonomy Climate Delegated Act, and what is the primary technical screening criterion it must meet?",
    "source_documents": ["2021_2139"],
    "relevant_section": "Annex I, Activity 4.3",
    "key_facts": [
      "Activity 4.3 covers electricity generation from wind power including offshore",
      "Wind power is considered to substantially contribute to climate change mitigation without a lifecycle GHG threshold",
      "DNSH criteria apply across all six environmental objectives",
      "The activity must comply with the minimum social safeguards in Article 18"
    ],
    "notes": "Tests ability to locate a specific annex activity and summarise its criteria"
  },
  {
    "id": "q2",
    "question": "What does 'Do No Significant Harm' to biodiversity mean for a solar photovoltaic electricity generation project under the EU Taxonomy?",
    "source_documents": ["2021_2139"],
    "relevant_section": "Annex I, Activity 4.1, DNSH criterion 6",
    "key_facts": [
      "Activity 4.1 covers electricity generation from solar photovoltaic technology",
      "DNSH to biodiversity requires an environmental impact assessment where required by law",
      "For projects in or near biodiversity-sensitive areas additional requirements apply",
      "The activity must not lead to significant deterioration of protected habitats"
    ],
    "notes": "Tests ability to navigate to a specific DNSH sub-criterion within a long annex"
  },
  {
    "id": "q3",
    "question": "What are the minimum safeguards that all EU Taxonomy-aligned activities must comply with regardless of their technical screening criteria?",
    "source_documents": ["2020_852"],
    "relevant_section": "Article 18",
    "key_facts": [
      "Article 18 of Regulation 2020/852 defines minimum safeguards",
      "Alignment with OECD Guidelines for Multinational Enterprises is required",
      "Alignment with UN Guiding Principles on Business and Human Rights is required",
      "The ILO core labour conventions must be respected",
      "Minimum safeguards apply in addition to the technical screening criteria"
    ],
    "notes": "Tests cross-document reasoning — minimum safeguards are in the main Regulation, not the Delegated Act"
  }
]
```

---

### `eval/harness.py`

```python
@dataclass
class EvalResult:
    question_id: str
    question: str
    answer: str
    sources_retrieved: list[dict]
    key_facts_found: list[str]      # which key_facts appear in the answer
    key_facts_missing: list[str]
    recall_score: float             # len(found) / len(all key_facts)
    notes: str

class EvaluationHarness:
    def __init__(self, dataset_path: str, rag_pipeline): ...

    def run(self) -> list[EvalResult]: ...
        # For each question in golden_dataset.json:
        # 1. Run rag_pipeline.query(question)
        # 2. Check which key_facts appear in the answer (simple substring check)
        # 3. Return EvalResult

    def report(self, results: list[EvalResult]) -> None: ...
        # Print a readable comparison table to stdout
```

The key_facts check is intentionally simple (substring / keyword presence). The goal is not to have sophisticated LLM-as-judge systems, but to make human eval easy. Details can be added to `metrics.py` later.

---

### `CLAUDE.md`

This file gives instructions to future AI coding sessions. Include:

```markdown
# EU Taxonomy RAG — AI Session Instructions

## Project purpose
Experimental RAG platform for EU Taxonomy documents. Solo research project.
The developer experiments with different ingestion strategies, RAG pipelines,
and agent designs. Prioritise clean interfaces and swappability over cleverness.

## Key conventions
- Package lives in src/taxonomy_rag/
- Config is always imported from taxonomy_rag.config (settings singleton)
- DB layer: psycopg3 raw SQL, no ORM. Repository pattern.
- Ingestion strategies implement the Chunker or DocumentParser Protocol.
  Add new strategies as new files — never modify existing ones.
- RAG pipelines in rag/ all expose .ingest() and .query() with the same signature.
- Evaluation: run scripts/evaluate.py to benchmark against eval/golden_dataset.json.

## What not to do
- Do not add an ORM (SQLAlchemy etc.) — raw psycopg is intentional.
- Do not change the DB schema in 00_init.sql without flagging it — the vector
  dimension (384) is coupled to the embedding model choice.
- Do not implement ingestion parsers or chunkers without being asked — these are
  the core experiments and the developer fills them in deliberately.

## Running the project
docker compose up -d          # start Postgres
uv run python scripts/ingest.py data/raw/2021_2139_EN.pdf
uv run python scripts/evaluate.py
```

---

### Notebooks

Each notebook should contain only:
1. A markdown title cell explaining the notebook's purpose
2. A setup cell (load_dotenv, imports, db connection, print document count)
3. Section header markdown cells outlining what experiments will go in each section

Do not pre-fill experiment cells — the developer will write these.

---

## Reference implementation

The DB layer (`connection.py`, `repository.py`), embedding layer (`embedder.py`),
LLM layer (`provider.py`), and all three RAG pipelines (`naive.py`, `hybrid.py`,
`advanced.py`) should be **directly adapted** from the following reference project:

**Reference:** `rag-playground` — a learning project the developer completed before
starting this one. The patterns there are intentional and should be preserved:
- `@lru_cache` on model loading
- `with get_pool().connection() as conn:` pattern for all DB access
- `dict_row` row factory for read operations
- `Jsonb()` wrapper for metadata inserts
- `normalize_embeddings=True` on all embed calls
- The hybrid search SQL (RRF with k=60, pre_k = top_k * 4) in `repository.py`
- The HyDE + cross-encoder reranking pattern in `advanced.py`

The only changes from the reference:
- Package name `taxonomy_rag` instead of `rag_playground`
- `config.py` has additional ingestion settings (`default_chunk_size` etc.)
- `repository.py` `get_all()` should support filtering by metadata key:
  `def get_all(self, limit, offset, metadata_filter: dict | None = None)`

---

## What the scaffold should NOT include

- Any real PDF parsing logic — `pdf.py` raises `NotImplementedError`
- Any real chunking logic — all chunker `chunk()` methods raise `NotImplementedError`
- Any real agent logic — `ComplianceCheckerAgent` raises `NotImplementedError`
- Any LLM evaluation logic — `metrics.py` functions raise `NotImplementedError`
- Sample data or downloaded PDFs
- Any CI/CD configuration
- Any authentication

---

## Definition of done for the scaffold

The scaffold is complete when:
1. `docker compose up -d` starts Postgres cleanly
2. `uv sync` installs all dependencies without errors
3. `uv run python -c "from taxonomy_rag.config import settings; print(settings.dsn)"` works
4. `uv run python -c "from taxonomy_rag.db.connection import get_pool; get_pool()"` connects
5. All three notebooks open in Jupyter with the setup cell running without import errors
6. `uv run python scripts/evaluate.py` runs and exits gracefully (with a
   "no documents ingested yet" message, not a crash)
7. Every stub raises `NotImplementedError` with a descriptive message indicating
   what the developer should implement there
