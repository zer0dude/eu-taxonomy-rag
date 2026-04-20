# EU Taxonomy RAG — Working Notes

## Architecture

The system is built in strict layers. Each layer depends only on the layers below it.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 7 — Agents          agents/                                  │
│  AgentProtocol, AgentLoop, per-agent composition                    │
│  react, react_naive_corpus_vector_rag, react_naive_corpus_hybrid_rag│
│  react_naive_corpus_advanced_rag, react_naive_corpus_multi_rag      │
│  llm_direct, mock                                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 6 — Agent tools     tools/                                   │
│  Tool protocol, ToolKit, SearchCorpusTool, ReadFullDocument         │
│  tools/base.py  tools/search/corpus.py  tools/attachment/read_full │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 5 — Retrieval       retrieval/                               │
│  Retriever protocol, RetrievalResult, CorpusScope                   │
│  NaiveVectorRetrieval, HybridRetrieval, AdvancedRetrieval           │
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
- `reference/rag/` — prior-project code kept as read-only reference (now implemented as `HybridRetrieval` and `AdvancedRetrieval` in `retrieval/`)

---

## Agent model

Each agent is defined by three explicit components:

```
┌───────────────────────── Agent ────────────────────────────────────┐
│  Core loop    AgentLoop (llm/loop.py)                              │
│  Default prompt  prompts/<name>.txt  (baked in; --prompt overrides)│
│  ToolKit:                                                          │
│    ┌── DB retrieval tool ──────────────────────────────────────┐   │
│    │  SearchCorpusTool(retrieval, scope, name, description, tracer)│  │
│    └───────────────────────────────────────────────────────────┘   │
│    ┌── Attachment tool (when question has attachments) ─────────┐  │
│    │  ReadFullDocument(path_map, registry: ReaderRegistry)      │  │
│    └───────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

**Adding a new retrieval method:** implement `Retriever` in `retrieval/`, pass it to
`SearchCorpusTool`, register a new agent in `agents/`. No other changes needed.

**Adding a new agent tool:** implement `Tool` protocol in `tools/`, add to ToolKit in the agent.

**Naming convention:** `react_{ingestion}_{corpus}_{retrieval}_rag`
- `naive_corpus` = naively-chunked `naive_pdf` ingestion strategy
- `vector` / `hybrid` / `advanced` / `multi` = retrieval method
- Tool names follow: `search_{ingestion}_{corpus}_{retrieval}`
- Corpus constant: `NAIVE_CORPUS` (= `CorpusScope(ingestion_strategy="naive_pdf")`)

**SearchCorpusTool** — `name`, `description`, and `tracer` are constructor params (not class constants).
Rebuild the tool per `answer()` call to inject the current tracer. Retriever instances are stateless
and can be constructed once in `__init__`.

**Current agents:**

| Agent | Loop | DB tool(s) | Attachment | Default prompt |
|---|---|---|---|---|
| `mock` | none | none | none | none |
| `llm_direct` | none | none | none | none |
| `react` | ReAct | none | ReadFullDocument | none |
| `react_naive_corpus_vector_rag` | ReAct | `search_naive_corpus_vector` (NaiveVectorRetrieval) | ReadFullDocument | compliance_v1.txt |
| `react_naive_corpus_hybrid_rag` | ReAct | `search_naive_corpus_hybrid` (HybridRetrieval) | ReadFullDocument | compliance_v1.txt |
| `react_naive_corpus_advanced_rag` | ReAct | `search_naive_corpus_advanced` (AdvancedRetrieval) | ReadFullDocument | compliance_v1.txt |
| `react_naive_corpus_multi_rag` | ReAct | `search_naive_corpus_hybrid` + `search_naive_corpus_advanced` | ReadFullDocument | compliance_v1.txt |

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

**8. Last-iteration warning must be aligned with the compliance prompt.**
The loop injects a `[Final research step]` user message before the last LLM call, and
makes one post-loop synthesis call if the model still makes tool calls. Neither is enough
alone: `compliance_v1.txt` also has a "Research time limits" section establishing that a
`PRELIMINARY — research incomplete` answer is explicit policy. Prompt and loop must agree
or the stricter instruction wins and the agent still refuses. Token usage
(`prompt_tokens` / `completion_tokens`) is now captured from every `litellm.completion()`
call via `tracer.log_usage()` and surfaced per-iteration in traces, per-question in
`outcomes.csv`, and as run totals in `metadata.json`. Ollama may return zeros.

**11. HTML report is the right eval review format.**
`evaluate.py` generates a self-contained `report.html` per run alongside metadata.json and
outcomes.csv. The HTML embeds marked.js (v15, ~39 KB inline) so agent markdown answers render
correctly in any browser with no internet access. Reasoning is collapsible via native `<details>`.
Evaluator scores and notes are persisted to `localStorage` keyed by `run_id + question_id` — no
server needed, survives page reload. `scripts/report.py` regenerates the HTML for any existing
run dir. Default `--log-level` is now `full` so tool results are captured in traces automatically.

**9. SearchCorpusTool must be rebuilt per answer() call to carry the tracer.**
`SearchCorpusTool` accepts `name`, `description`, and `tracer` as constructor params.
Because HyDE (`AdvancedRetrieval`) calls the LLM inside `retrieve()`, it needs the
per-question tracer to log those tokens correctly. Build the tool in `answer()` (not
`__init__`); the underlying retriever instance is stateless and can be shared across calls.
HyDE tokens are logged at iteration index 0, which is unused by `AgentLoop` (1-indexed),
so they appear cleanly as a "pre-retrieval" entry in traces and accumulate in totals.

**10. Corpus scope name must encode the ingestion method, not just "corpus".**
As more ingestion strategies are added (hierarchical, sentence-level, etc.), the corpus
constant and tool name must distinguish them. Convention: `NAIVE_CORPUS` / `search_naive_corpus_*`.
Future: `HIERARCHICAL_CORPUS` / `search_hierarchical_corpus_*`. The `CorpusScope` filter
still uses `ingestion_strategy="naive_pdf"` internally — the naming is for human/agent clarity.

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
| 2026-04-19 | react_naive_corpus_vector_rag | simple_v1 | 10/10 — v1.1 timeout fix confirmed |
| 2026-04-19 | react_naive_corpus_hybrid_rag | simple_v1 | 10/10 — new hybrid agent |
| 2026-04-19 | react_naive_corpus_advanced_rag | simple_v1 | 10/10 — new advanced agent (HyDE+rerank) |
| 2026-04-19 | react_naive_corpus_multi_rag | simple_v1 | 10/10 — agent used both tools |

Root cause of timeouts: compliance prompt + low iteration cap. See learning #2 above. Fixed in v1.1 — see learning #8 below.

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

## Next steps

1. ~~**Fix compliance prompt**~~ — done (v1.1): last-iteration warning + prompt fallback section
2. ~~**Re-run hard_01 eval**~~ — done (v1.1): timeouts resolved, preliminary answers labelled
3. ~~**HybridRetrieval + AdvancedRetrieval**~~ — done: `retrieval/hybrid.py`, `retrieval/advanced.py`, 3 new agents + multi-tool agent; consistent naming scheme; HyDE tokens tracked at iteration 0
4. ~~**Rework eval output**~~ — done: `report.html` generated automatically per run; `scripts/report.py` regenerates for old runs. Markdown rendered, reasoning collapsible, notes persisted in localStorage, export to JSON.
5. **Comparative eval on hard_01** — run all four RAG agents on hard_01; compare retrieval quality and token cost; HyDE overhead vs. accuracy gain.
6. **Grounded reasoning eval** — confirm agent cites document sections; do not rely on training memory for regulatory claims.

---

## Ideas (not yet scoped)

- **Realistic / messy eval sets** — questions with too much information, irrelevant documents, mixed attachment types. Naturally coupled with attachment reading improvements below.
- **Alternative ingestion strategies** — hierarchical chunking, structural chunking, and (distinct category) manually authored natural-language descriptions of graphs so graph content becomes retrievable and LLM-interpretable.
- **More targeted DB search tools** — let agents scope searches below the full corpus level (by document, section, or type). `CorpusScope` already supports this pattern; new ingestion strategies will need it.
- **Richer attachment reading** — readers for URLs, spreadsheets, slide decks; tooling for directed partial reads rather than full-document reads only.
