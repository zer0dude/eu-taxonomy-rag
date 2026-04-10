# EU Taxonomy RAG

A research and experimentation platform for applying **Retrieval-Augmented Generation (RAG)** and **LangChain agents** to the **EU Taxonomy for Sustainable Finance**.

The EU Taxonomy (Regulation 2020/852 and its Climate Delegated Act 2021/2139) is a classification system that defines which economic activities qualify as environmentally sustainable. Its documents are long, dense, and full of cross-references — a good test bed for RAG systems.

The goal is not a production system. It is a learning platform: different ingestion strategies, retrieval methods, and agent designs are swapped in and out to compare their behaviour on the same domain.

## Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Package manager | `uv` |
| Database | PostgreSQL 16 + pgvector |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dims) |
| LLM abstraction | LangChain with swappable providers (Ollama / Anthropic / OpenAI) |
| PDF parsing | `pymupdf` (baseline) |

## Getting started

```bash
# 1. Copy and fill in config
cp .env.example .env

# 2. Start Postgres
docker compose up -d

# 3. Install dependencies
uv sync --extra dev

# 4. Verify the stack
uv run python -c "from taxonomy_rag.config import settings; print(settings.dsn)"

# 5. Ingest a document (parsers must be implemented first)
uv run python scripts/ingest.py data/raw/2021_2139_EN.pdf

# 6. Run evaluation
uv run python scripts/evaluate.py
```

## Project structure

```
src/taxonomy_rag/
├── config.py          # pydantic-settings, reads from .env
├── db/                # connection pool, DocumentRepository (raw SQL)
├── embeddings/        # Embedder wrapper around sentence-transformers
├── llm/               # get_llm() factory (Ollama / Anthropic / OpenAI)
├── ingestion/         # parsers, chunkers, pipeline orchestrator
├── rag/               # NaiveRAG, HybridRAG, AdvancedRAG
├── agents/            # ComplianceCheckerAgent + LangChain tools
└── eval/              # EvaluationHarness, metrics

scripts/
├── ingest.py          # CLI: ingest a document
└── evaluate.py        # CLI: benchmark against golden dataset

eval/
└── golden_dataset.json  # ground-truth Q&A pairs

notebooks/
├── 01_ingestion_experiments.ipynb
├── 02_retrieval_experiments.ipynb
└── 03_agent_experiments.ipynb
```

## Experiments

See [notes.md](notes.md) for the current experimental plan, status, and learnings.

The three RAG pipelines share an identical `.query()` interface so they can be swapped into the same evaluation loop without changing anything else.

## Source documents

Drop EU Taxonomy PDFs into `data/raw/`. They are gitignored and never committed.
Key documents:
- `2020_852` — Regulation (EU) 2020/852 (the main Taxonomy Regulation)
- `2021_2139` — Climate Delegated Act (technical screening criteria)
