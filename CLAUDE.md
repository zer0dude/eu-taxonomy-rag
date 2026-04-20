# EU Taxonomy RAG — AI Session Instructions

## Project purpose

Experimental RAG platform for EU Taxonomy documents. Solo research project.
The developer experiments with different ingestion strategies, retrieval methods,
and agent designs. Prioritise clean interfaces and swappability over cleverness.

Read `notes.md` at the start of a session to get oriented quickly.

After the user greenlights tests, update `notes.md` (new learnings, revised next steps)
**before** committing. Do not commit without a notes update when behaviour or architecture changed.
Do not commit unless the user gives greenlights.

---

## Architecture overview

The system is built in strict layers. Each layer depends only on the layers below it.

```
Layer 7  agents/           AgentProtocol, per-agent composition
Layer 6  tools/            Tool protocol, ToolKit, SearchCorpusTool, ReadFullDocument
Layer 5  retrieval/        Retriever protocol, NaiveVectorRetrieval, CorpusScope
Layer 4  db/               DocumentRepository (raw psycopg3, no ORM)
Layer 3  ingestion/        strategies/ → parsers/ + chunkers/ → IngestionPipeline
Layer 2  readers/          AttachmentReader protocol, ReaderRegistry, PDFReader
Layer 1  embeddings/ llm/ config.py tracing/   (foundation, no project imports)
```

`reference/rag/` — prior-project code kept as read-only reference for implementing
`retrieval/hybrid.py` and `retrieval/advanced.py`. Not imported by active code.

---

## Key conventions

- **Package:** `src/taxonomy_rag/`
- **Config:** always `from taxonomy_rag.config import settings` — never instantiate `Settings` directly
- **DB:** raw psycopg3, repository pattern in `db/repository.py`. No ORM.
- **Ingestion strategies:** implement `IngestionStrategy` protocol and add one line to `ingestion/strategies/registry.py`. Never modify existing strategies.
- **Chunkers / parsers:** add new files; never modify existing ones. Unimplemented strategies simply don't exist as files yet — no stubs.
- **Retrieval:** implement the `Retriever` protocol in `retrieval/`. Wrap with `SearchCorpusTool` to expose as an agent tool.
- **Agent tools:** implement the `Tool` protocol in `tools/`. Add to `ToolKit` in the relevant agent.
- **Agents:** each agent file exposes `get_agent()` at module level. `evaluate.py` auto-discovers by name.
- **Row factory:** every `DocumentRepository` method must set its own `row_factory` at the top of its connection block. Never rely on pool state.

---

## What not to do

- Do not add an ORM — raw psycopg3 is intentional.
- Do not change `docker/postgres/init/00_init.sql` without flagging it — `vector(384)` is coupled to `all-MiniLM-L6-v2`.
- Do not add stubs that raise `NotImplementedError` — if something is not built yet, the file simply does not exist.
- Do not implement chunkers or parsers without being asked — these are deliberate experiments.
- Do not put retrieval logic in the `rag/` module — it no longer exists in `src/`. Retrieval belongs in `retrieval/`.

---

## Running the project

```bash
docker compose up -d                          # start Postgres (+ pgAdmin on port 5050)
uv run python scripts/ingest.py data/raw/2021_2139_EN.pdf --strategy naive_pdf
uv run python scripts/ingest_corpus.py --strategy naive_pdf
uv run python scripts/evaluate.py \
    --questions eval/simple_v1/questions.json \
    --prompt prompts/compliance_v1.txt \
    --agent react_naive_rag
uv run pytest tests/unit/ -q                  # unit tests (no Docker needed)
uv run pytest tests/integration/ -q          # integration tests (Docker required)
```

---

## LLM provider selection

Set `LLM_PROVIDER` in `.env` to `anthropic`, `openai`, or `ollama`.
`litellm.completion()` is the only LLM call site — no provider-specific code in agents.
