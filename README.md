# EU Taxonomy RAG

A research and experimentation platform for applying **Retrieval-Augmented Generation (RAG)**
and **LLM agents** to the **EU Taxonomy for Sustainable Finance**.

The EU Taxonomy (Regulation 2020/852 and its Climate Delegated Act 2021/2139) is a
classification system that defines which economic activities qualify as environmentally
sustainable. Its documents are long, dense, and full of cross-references — a demanding
test bed for RAG systems.

The goal is not a production system. It is a learning platform: ingestion strategies,
retrieval methods, and agent designs are swapped in and out to compare their behaviour on
the same evaluation set.

---

## Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Package manager | `uv` |
| Database | PostgreSQL 16 + pgvector (HNSW index, cosine distance) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384 dims) |
| LLM abstraction | LiteLLM — provider-agnostic, OpenAI wire format |
| LLM providers | Ollama (local) / Anthropic / OpenAI — selected via `.env` |
| PDF parsing | `pymupdf` (fitz) |
| Config | `pydantic-settings`, reads from `.env` |
| Containerisation | Docker Compose — Postgres + pgAdmin; Python runs on the host |

---

## Getting started

```bash
# 1. Copy and fill in config
cp .env.example .env

# 2. Start Postgres and pgAdmin
docker compose up -d

# 3. Install dependencies
uv sync --extra dev

# 4. Verify the stack
uv run python -c "from taxonomy_rag.config import settings; print(settings.dsn)"

# 5. Ingest documents
uv run python scripts/ingest_corpus.py --strategy naive_pdf

# 6. Run evaluation
uv run python scripts/evaluate.py \
    --questions eval/simple_v1/questions.json \
    --agent react_naive_rag
```

pgAdmin is available at http://localhost:5050 after `docker compose up -d pgadmin`
(`admin@local.dev` / `admin`). Connect to host `postgres`, port `5432`, db/user/password `taxonomy`.

---

## Project structure

```
src/taxonomy_rag/
├── config.py              pydantic-settings singleton, reads from .env
├── db/                    DocumentRepository — raw psycopg3, no ORM
│   ├── connection.py      connection pool (psycopg_pool)
│   └── repository.py      insert, vector_search, hybrid_search, get_all
├── embeddings/            Embedder wrapper around sentence-transformers
├── llm/
│   ├── loop.py            AgentLoop — stateless, reusable tool-use loop
│   └── provider.py        LiteLLM model string + completion kwargs per provider
├── ingestion/
│   ├── pipeline.py        IngestionPipeline orchestrator (parse → chunk → embed → store)
│   ├── models.py          ParsedDocument, Chunk dataclasses
│   ├── parsers/           PDFParser — extracts per-page text + EU taxonomy metadata
│   ├── chunkers/          NaiveChunker — sliding word-window with overlap
│   └── strategies/        NaivePDFStrategy, StrategyRegistry
├── readers/               PDFReader, ReaderRegistry — raw text extraction for agent tools
├── retrieval/
│   ├── base.py            Retriever protocol, RetrievalResult dataclass
│   ├── scope.py           CorpusScope, NAIVE_PDF_CORPUS
│   └── naive.py           NaiveVectorRetrieval — embed → cosine search
├── tools/
│   ├── base.py            Tool protocol, ToolKit
│   ├── search/corpus.py   SearchCorpusTool(retrieval, scope)
│   └── attachment/        ReadFullDocument — reads per-question attachment files
├── agents/
│   ├── base.py            AgentProtocol
│   ├── mock.py            Hardcoded response — validates pipeline plumbing
│   ├── llm_direct.py      Bare LLM, no tools — training-memory baseline
│   ├── react.py           ReAct with attachment reading only
│   └── react_naive_rag.py ReAct + corpus search + attachment reading
└── tracing/               Tracer protocol, FileTracer, NullTracer

scripts/
├── ingest.py              CLI: ingest a single file
├── ingest_corpus.py       CLI: ingest all PDFs in data/raw/
└── evaluate.py            CLI: run an agent over a question set, save to runs/

eval/
├── simple_v1/             10 manufacturing eligibility questions
├── golden_dataset_v1/     3 baseline wind/solar/safeguards questions
└── hard_01/               3 adversarial energy questions with PDF attachments

prompts/
└── compliance_v1.txt      Grounded-citation system prompt (default for react_naive_rag)

reference/
└── rag/                   Prior-project pipeline code — reference only, not imported
```

---

## How evaluation works

`scripts/evaluate.py` has three interchangeable axes:

```bash
uv run python scripts/evaluate.py \
    --questions eval/simple_v1/questions.json \   # any question set
    --prompt prompts/compliance_v1.txt \           # optional; agent uses its default if omitted
    --agent react_naive_rag \                      # any agent in src/taxonomy_rag/agents/
    --log-level truncated                          # full | truncated | metadata | none
```

Each run saves to `runs/{timestamp}_{agent}_{question_set}/`:
- `metadata.json` — run parameters and timing
- `outcomes.csv` — one row per question; fill `human_score` / `human_notes` manually
- `traces/` — per-question structured trace (reasoning + tool calls)

Adding a new agent: create `src/taxonomy_rag/agents/my_agent.py` with a `get_agent()` function.
The eval script auto-discovers it by name — no changes to `evaluate.py` needed.

---

## Source documents

Drop EU Taxonomy PDFs into `data/raw/`. They are gitignored and never committed.

Organise by type for automatic `document_type` inference:
```
data/raw/
├── core_regulation/                 → document_type: "regulation"
├── delegated_acts_technical_criteria/ → document_type: "delegated_act"
├── guidance_documents/              → document_type: "guidance"
└── comission_notices_interpretive_guidance_faqs/ → document_type: "notice"
```

Key documents:
- `2020_852` — Regulation (EU) 2020/852 (the main Taxonomy Regulation)
- `2021_2139` — Climate Delegated Act (technical screening criteria)

See [notes.md](notes.md) for the current experimental plan, key learnings, and next steps.
