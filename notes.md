# EU Taxonomy RAG — Working Notes

## Architecture

The system is built in strict layers. Each layer depends only on the layers below it.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 7 — Agents          agents/                                  │
│  AgentProtocol, AgentLoop, per-agent composition                    │
│  react, react_naive_rag, llm_direct, mock                           │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 6 — Agent tools     tools/                                   │
│  Tool protocol, ToolKit, SearchCorpusTool, ReadFullDocument         │
│  tools/base.py  tools/search/corpus.py  tools/attachment/read_full │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 5 — Retrieval       retrieval/                               │
│  Retriever protocol, RetrievalResult, CorpusScope                   │
│  NaiveVectorRetrieval → future: HybridRetrieval, AdvancedRetrieval  │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 4 — DB              db/                                      │
│  DocumentRepository — insert, vector_search, hybrid_search, get_all│
│  Raw psycopg3, no ORM. vector(384) via pgvector HNSW index.         │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3 — Ingestion       ingestion/                               │
│  strategies/ — named recipes (NaivePDFStrategy)                     │
│    └─ build_pipeline() → IngestionPipeline(parser, chunker,         │
│                          embedder, repo, strategy_name)             │
│  parsers/  — PDFParser → ParsedDocument (per-page text + metadata)  │
│  chunkers/ — NaiveChunker → list[Chunk]                             │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2 — Extractors      readers/                                 │
│  AttachmentReader protocol, ReaderRegistry                          │
│  PDFReader delegates to PDFParser for text; used by ReadFullDocument│
├─────────────────────────────────────────────────────────────────────┤
│  Layer 1 — Foundation      embeddings/  llm/  config.py  tracing/  │
│  Embedder (sentence-transformers), AgentLoop (litellm), Settings,   │
│  Tracer protocol, FileTracer, NullTracer                            │
└─────────────────────────────────────────────────────────────────────┘
```

**Cross-cutting concerns:**
- `config.py` — `settings` singleton, read everywhere via `from taxonomy_rag.config import settings`
- `tracing/` — `Tracer` protocol, `FileTracer`, `NullTracer`; passed through every agent call
- `reference/rag/` — prior-project code kept as reference for implementing `HybridRetrieval` and `AdvancedRetrieval`

---

## Agent model

Each agent is defined by three explicit components:

```
┌───────────────────────── Agent ────────────────────────────────────┐
│  Core loop    AgentLoop (llm/loop.py)                              │
│  Default prompt  prompts/<name>.txt  (baked in; --prompt overrides)│
│  ToolKit:                                                          │
│    ┌── DB retrieval tool ──────────────────────────────────────┐   │
│    │  SearchCorpusTool(retrieval: Retriever, scope: CorpusScope)│   │
│    └───────────────────────────────────────────────────────────┘   │
│    ┌── Attachment tool (when question has attachments) ─────────┐  │
│    │  ReadFullDocument(path_map, registry: ReaderRegistry)      │  │
│    └───────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

**Adding a new retrieval method:** implement `Retriever` in `retrieval/`, pass it to
`SearchCorpusTool`, register a new agent in `agents/`. No other changes needed.

**Adding a new agent tool:** implement `Tool` protocol in `tools/`, add to ToolKit in the agent.

**Current agents:**

| Agent | Loop | DB tool | Attachment | Default prompt |
|---|---|---|---|---|
| `mock` | none | none | none | none |
| `llm_direct` | none | none | none | none |
| `react` | ReAct | none | ReadFullDocument | none |
| `react_naive_rag` | ReAct | SearchCorpusTool (NaiveVectorRetrieval + NAIVE_PDF_CORPUS) | ReadFullDocument | compliance_v1.txt |

---

## Key learnings

**1. RAG value is grounded citations, not factual correctness.**
Even without RAG, the LLM answers EU Taxonomy questions correctly from training memory
(confirmed: react on hard_01 — 3/3 correct with no retrieval). The value of RAG is
producing audit-quality output that cites specific document sections. That is the target.

**2. Compliance prompt + iteration cap interact badly.**
The strict "do not answer without retrieved evidence" prompt causes timeout loops when
retrieval fails to find a conclusive chunk. The agent repeats the same search until it
hits the cap and returns a sentinel. Fix: add a fallback instruction — "after 5 failed
searches, synthesise the best answer from what you have, stating what is missing."

**3. LiteLLM is the right abstraction layer.**
`litellm.completion()` uses the OpenAI message/tool format as canonical wire format,
works identically across Anthropic, OpenAI, and Ollama. No provider-specific code paths
in agent logic. Migration from LangChain: removed 5 packages, gained clarity.

**4. AgentLoop is stateless and reusable.**
All ReAct-style agents share one loop class (`llm/loop.py`). Sub-agents can be
registered as ordinary tools in ToolKit — the "agents-as-tools" pattern requires no
special framework. Implementation details: `message.content` is `None` when
`finish_reason == "tool_calls"` (guard with `or ""`); `call.function.arguments` is a
JSON string (always `json.loads()` before passing to toolkit).

**5. Strategy tagging enables A/B retrieval comparison.**
Every chunk stores `ingestion_strategy` in its JSONB metadata. `CorpusScope` filters on
this field via the GIN index. Two ingestion strategies can coexist in the same DB and be
queried independently — no table isolation needed.

**6. DB row_factory must be set per method.**
`psycopg3` connection pools can return connections with a row_factory set by a prior
caller. Every repository method must set its own row_factory at the top of its connection
block (`dict_row` for reads, `tuple_row` for inserts). Never rely on pool default.

**7. Prompt ownership is per-agent, not per-run.**
Each agent class loads its default prompt at import time. `--prompt` on `evaluate.py`
overrides it for experimental runs. The override is recorded in `metadata.json`; omitting
`--prompt` records `null`. This means an agent without a default will silently use
`"You are a helpful assistant."` — design agents with explicit defaults.

---

## Eval infrastructure

**`scripts/evaluate.py`** — three interchangeable axes:
- `--questions` — any questions JSON (`eval/simple_v1/`, `eval/golden_dataset_v1/`, `eval/hard_01/`)
- `--prompt` — any file from `prompts/`; omit to use agent default
- `--agent` — any module in `src/taxonomy_rag/agents/` that exposes `get_agent()`
- `--log-level` — `full` / `truncated` / `metadata` (default) / `none`

Each run saves to `runs/{YYYYMMDD}_{HHMMSS}_{agent}_{question_set}/`:
- `metadata.json` — machine-readable run summary
- `outcomes.csv` — one row per question; fill `human_score` and `human_notes` manually
- `traces/{question_id}_trace.json` — structured per-question agent trace

**Eval results to date:**

| Run | Agent | Questions | Result |
|---|---|---|---|
| 2026-04-13 | react (no RAG) | hard_01 | 3/3 correct — strong baseline |
| 2026-04-18 | react_naive_rag | simple_v1 | 9/10 — 1 timeout (iteration cap) |
| 2026-04-18 | react_naive_rag | hard_01 | 1/3 — 2 timeouts (iteration cap) |

Root cause of timeouts: compliance prompt + low iteration cap. See learning #2 above.

---

## DB management (quick reference)

```sql
-- What's in the DB
SELECT metadata->>'ingestion_strategy' AS strategy,
       metadata->>'ingest_run_id'      AS run_id,
       COUNT(*)                         AS chunks,
       MIN(created_at)                  AS started,
       MAX(created_at)                  AS finished
FROM documents
GROUP BY 1, 2
ORDER BY started;

-- Inspect a document's chunks
SELECT id, metadata->>'page_range', LEFT(content, 120)
FROM documents
WHERE metadata->>'document_id' = '32021r2139'
  AND metadata->>'ingestion_strategy' = 'naive_pdf'
ORDER BY (metadata->>'chunk_index')::int
LIMIT 20;

-- Delete a specific run
DELETE FROM documents WHERE metadata->>'ingest_run_id' = '<uuid>';

-- Delete all chunks for one strategy
DELETE FROM documents WHERE metadata->>'ingestion_strategy' = 'naive_pdf';

-- Wipe everything
TRUNCATE documents;
```

pgAdmin: `docker compose up -d pgadmin` → http://localhost:5050 (`admin@local.dev` / `admin`).
Connect: host `postgres`, port `5432`, db/user/password `taxonomy`.

---

## Next steps (v1 baseline)

1. **Fix compliance prompt** — add fallback instruction after N failed searches
2. **HybridRetrieval** — implement in `retrieval/hybrid.py` using `reference/rag/hybrid.py` as reference; wire to a new `react_hybrid_rag` agent; compare against `react_naive_rag`
3. **Grounded reasoning eval** — confirm agent cites document sections; do not rely on training memory for regulatory claims
4. **hard_02 eval set** — realistic messy attachments (URLs, internal spreadsheets, incomplete data)
