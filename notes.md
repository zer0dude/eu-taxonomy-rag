# EU Taxonomy RAG — Working Notes

## Experimental plan

Build incrementally. Each step should be testable and produce a measurable result before moving on.

1. **Evaluation baseline** — refine golden questions, lock down scoring method
2. **Bare LLM baseline** — run golden questions through the LLM with no context or tools
   - Records the floor: what does the LLM already know from training?
3. **Agent tracing / run logs** — structured per-question log of what the agent did
   (tool calls, reasoning, iterations) so results can be inspected and compared
4. **PDF spike** — extract raw text from one real document.
5. **Naive ingestion** — implement `PDFParser` + `NaiveChunker`, ingest one document
6. **Naive RAG tool** — expose the vector DB as an agent tool; compare to bare-LLM floor
7. **Grounded reasoning** — constrain the agent to cite official documents for every claim;
   no relying on training memory for regulatory content
8. **Iterate on retrieval** — vary chunker, retrieval method (`HybridRAG`, `AdvancedRAG`), compare scores
9. **Harder eval sets** — realistic attachments: messy URLs, internal spreadsheets with
   incomplete data; questions where data is insufficient or contradictory

**Key principle:** the agent layer only adds value if the underlying tools work. Don't add reasoning complexity on top of broken retrieval.

---

## Current status

- [x] Evaluation baseline
- [x] Bare LLM baseline (llm_direct on simple_v1 + golden_dataset_v1)
- [x] ReAct agent with attachment tools (react on hard_01)
- [x] Agent tracing / run logs
- [x] PDF spike
- [x] Naive ingestion
- [ ] Naive RAG tool  ← next
- [ ] Grounded reasoning
- [ ] Retrieval iteration
- [ ] Harder eval sets

---

## Eval infrastructure (completed)

**scripts/evaluate.py** — modular eval script. Three interchangeable axes:
- `--questions` — any questions JSON (simple_v1, golden_dataset_v1, hard_01, ...)
- `--prompt` — any prompt file from `prompts/`
- `--agent` — any registered agent name; auto-discovered from `src/taxonomy_rag/agents/`

Each run saves to `runs/{YYYYMMDD}_{HHMMSS}_{agent}_{question_set}/`:
- `metadata.json` — machine-readable run summary (includes log_level used)
- `outcomes.csv` — one row per question; human fills `human_score` and `human_notes`
- `traces/{question_id}_trace.json` — structured per-question agent trace (see below)

**Question sets:**
- `eval/simple_v1/questions.json` — 10 manufacturing eligibility questions, adversarial
- `eval/golden_dataset_v1/questions.json` — 3 baseline wind/solar/safeguards questions
- `eval/hard_01/questions.json` — 3 adversarial energy questions with PDF attachments

**Agents:**
- `mock` — hardcoded fixed string; validates pipeline plumbing
- `llm_direct` — bare Anthropic API call, no tools, no retrieval
- `react` — Anthropic tool-use loop; reads attachments via `read_full_document` tool

**Tracing (`--log-level`):**
- `metadata` (default) — reasoning text + tool call names/inputs/sizes/timing; no content
- `truncated` — as above, plus first N chars of each tool result (`--truncate-chars`, default 500)
- `full` — complete tool result text stored in trace
- `none` — no trace files written
- Traces saved to `runs/{run_id}/traces/`; `NullTracer` used when log-level is none

**Prompts:**
- `prompts/base_v1.txt` — EU Taxonomy expert, eligibility vs alignment framing

---

## Learnings

**ReAct on hard_01 (2026-04-13):** The simple ReAct agent (Haiku 4.5, read_full_document
tool only) gave correct and precise answers on all three adversarial questions, with good
reasoning, despite having no access to the EU Taxonomy regulatory documents. The agent
read all relevant attachments in the first iteration (batching tool calls) and answered in
the second. This is a strong baseline — it also means the bar for "does RAG help?" is
higher than expected, since the LLM already knows the regulatory content from training.

**Key implication:** the value of RAG for this project is not just answering correctly — it
is *grounded reasoning with citations*. The model can get the right answer from memory; what
it cannot do from memory alone is produce audit-quality output that cites specific document
sections as its source. That is the real target.

**LangChain → LiteLLM migration (2026-04-16):** Replaced all LangChain usage with LiteLLM.
Decision rationale: LangChain is heavy, opinionated, and makes swapping providers harder
than it should be. LiteLLM is a thin normalization layer — `litellm.completion()` works
identically across Anthropic, OpenAI, and Ollama using the OpenAI message/tool format as
the canonical wire format. No provider-specific code paths in agent logic.

Key changes made:
- `pyproject.toml`: removed 5 langchain-* packages, added `litellm>=1.0`
- `config.py`: added `litellm_model_string` computed field (bare name for Anthropic,
  `ollama/{model}` prefix for Ollama, bare name for OpenAI)
- `llm/provider.py`: full rewrite — `get_model_string()` + `get_completion_kwargs()`
  with `lru_cache` (must call `.cache_clear()` between tests that patch settings)
- `tools/base.py`: `to_anthropic_schema()` renamed to `to_litellm_schema()`; output
  changed to OpenAI function-calling format `{"type":"function","function":{...}}`
- `agents/react.py`: simplified ~65 lines; delegates entirely to new `AgentLoop`
- `agents/llm_direct.py`: uses `litellm.completion()` instead of `anthropic` SDK
- `rag/naive.py`, `rag/hybrid.py`, `rag/advanced.py`: replaced `ChatPromptTemplate |
  get_llm()` chains with direct `litellm.completion()` calls
- `agents/tools/search.py`, `agents/tools/filters.py`: rewritten from `@tool` decorators
  to plain classes implementing the Tool protocol (bodies remain `NotImplementedError`)

**AgentLoop (2026-04-16):** Extracted the tool-use loop from react.py into a new shared
class `llm/loop.py`. `AgentLoop` is stateless and reusable — all future ReAct-style
agents and orchestrators share the same loop code. This is the foundation for multi-agent
setups: sub-agents can be registered as ordinary tools in `ToolKit`, exposed to an
orchestrator using the same `AgentLoop`. The "agents-as-tools" pattern requires no special
framework — just wrap an agent's `.answer()` call as a `Tool.run()`.

Critical implementation details to remember:
- `message.content` is `None` when `finish_reason == "tool_calls"` — always guard with `or ""`
- `call.function.arguments` is a JSON *string* — always `json.loads()` before passing to `toolkit.run()`
- Serialize assistant message back to dict via `message.model_dump()` (fall back to `dict(message)`)

**Tracing confirmed working post-migration (2026-04-16):** `AgentLoop` calls
`tracer.log_reasoning()` and `tracer.log_tool_call()` identically to the old react.py.
With Claude/Haiku, `message.content` is non-empty even during tool-call iterations
(the model explains its reasoning before calling tools), so structured traces are rich.
hard_01 re-run confirmed: same 2-iteration pattern, all attachments batched in iteration 1,
detailed answers in iteration 2. Architecture and eval quality unchanged by migration.

**Test suite established (2026-04-16):** Created full test suite from scratch (none existed
before). 59 tests total: 47 unit (no external services) + 12 integration (skip if no Docker).
- `tests/unit/test_provider.py` — model string + completion kwargs per provider
- `tests/unit/test_loop.py` — AgentLoop: text-only, tool calls, multiple tools, max iterations
- `tests/unit/test_toolkit.py` — to_litellm_schema format, run dispatch, error handling
- `tests/unit/test_rag.py` — NaiveRAG + HybridRAG with mocked repo + litellm
- `tests/unit/test_agents_unit.py` — MockAgent + LLMDirectAgent with mocked litellm
- `tests/integration/test_repository.py` — live DB CRUD, vector search, hybrid search, metadata filter

**Bug fixed: psycopg3 pool row_factory contamination (2026-04-16):** `get_by_id()` was
setting `conn.row_factory = dict_row` on a connection and returning it to the pool.
When `insert()` reused that connection, `fetchone()` returned a dict, and `row[0]` failed
with `KeyError: 0`. Fix: added explicit `conn.row_factory = tuple_row` at the start of
`insert()`. Pattern to follow: every method that changes `row_factory` must do so at the
top of its own connection block.

---

## Naive ingestion (2026-04-17)

Implemented `PDFParser` (pymupdf `page.get_text()`) and `NaiveChunker` (sliding
word-window, default 512 words / 50 overlap). Both live in
`src/taxonomy_rag/ingestion/parsers/pdf.py` and
`src/taxonomy_rag/ingestion/chunkers/naive.py`.

`document_id` is derived from the filename (`CELEX_32021R2139_EN_TXT.pdf` →
`32021r2139`); `document_type` is inferred from the parent directory name
(`delegated_acts_technical_criteria` → `delegated_act`, etc.).

Full corpus ingested via `scripts/ingest_corpus.py` (14 PDFs). Single-file
ingestion still available via the existing `scripts/ingest.py`.

39 unit tests added in `tests/unit/test_ingestion.py` — all passing.

**Infrastructure:** pgAdmin added to `docker-compose.yml`. After `docker compose
up -d pgadmin`, the UI is available at http://localhost:5050
(`admin@local.dev` / `admin`). Connect to host `postgres`, port `5432`,
db/user/password all `taxonomy`. Useful for browsing chunks and running ad-hoc
SQL against the documents table.

---

## Ingestion strategy architecture (2026-04-17)

Added a `strategies/` layer so that new ingestion recipes can be added without
touching any existing code.

**Layers:**
- `src/taxonomy_rag/ingestion/parsers/` — parse primitives (PDFParser, etc.)
- `src/taxonomy_rag/ingestion/chunkers/` — chunk primitives (NaiveChunker, etc.)
- `src/taxonomy_rag/ingestion/strategies/` — named compositions (NaivePDFStrategy, ...)
- `src/taxonomy_rag/ingestion/strategies/registry.py` — `DEFAULT_REGISTRY` (one line to add a strategy)
- `scripts/ingest.py` / `scripts/ingest_corpus.py` — CLI, never changes when new strategies are added

**Adding a new strategy:** create a new file in `strategies/`, then add one line to
`DEFAULT_REGISTRY` in `registry.py`. No script or pipeline changes needed.

**Every chunk now stores:**
```
ingestion_strategy  — e.g. "naive_pdf"
ingest_run_id       — UUID per corpus run (printed by ingest_corpus.py)
chunk_strategy      — chunker name e.g. "naive"
document_id, document_type, source, chunk_index, page_range
```

**Filtered search:** `DocumentRepository.vector_search()` and `hybrid_search()` now
accept an optional `metadata_filter` dict (JSONB `@>` operator, uses GIN index).
`NaiveRAG(ingestion_strategy="naive_pdf")` scopes all retrieval to that strategy.

28 new unit tests in `tests/unit/test_strategies.py` — 67 unit tests total.

**Action needed before next ingest:** the corpus was first ingested without
`ingestion_strategy` / `ingest_run_id`. Run `TRUNCATE documents;` in pgAdmin or psql,
then re-ingest with `scripts/ingest_corpus.py --strategy naive_pdf`.

**Note on scripts/:** `scripts/` contains CLI entry points (`ingest.py`,
`ingest_corpus.py`, `evaluate.py`) — thin orchestration wrappers over library code
in `src/`. One-off exploratory scripts (e.g. `spike_pdf.py`) do not belong here and
should be deleted once their purpose is served.

---

## DB management

**Inspect what is in the DB** (pgAdmin Query Tool or psql):
```sql
SELECT metadata->>'ingestion_strategy' AS strategy,
       metadata->>'ingest_run_id'      AS run_id,
       COUNT(*)                         AS chunks,
       MIN(created_at)                  AS started,
       MAX(created_at)                  AS finished
FROM documents
GROUP BY 1, 2
ORDER BY started;
```

**Delete a specific run** (re-ingest after a bug fix):
```sql
DELETE FROM documents WHERE metadata->>'ingest_run_id' = '<uuid>';
```

**Delete all chunks for one strategy:**
```sql
DELETE FROM documents WHERE metadata->>'ingestion_strategy' = 'naive_pdf';
```

**Delete legacy untagged chunks** (ingested before strategy tracking):
```sql
DELETE FROM documents WHERE metadata->>'ingestion_strategy' IS NULL;
```

**Wipe everything and start fresh:**
```sql
TRUNCATE documents;
```

**Inspect a specific document's chunks in order:**
```sql
SELECT id, metadata->>'page_range', LEFT(content, 120)
FROM documents
WHERE metadata->>'document_id' = '32021r2139'
  AND metadata->>'ingestion_strategy' = 'naive_pdf'
ORDER BY (metadata->>'chunk_index')::int
LIMIT 20;
```

---

## Open questions

- How do we force the agent to ground claims in retrieved documents rather than training
  memory? Prompt constraint? Tool design (e.g. tool that returns article text forces a
  citation)?
- For the RAG DB tool: should the agent get raw chunk text, or structured results
  (article reference + text)? Structured results make citations easier.
- What does a "realistic" messy attachment look like for hard_02? Candidates:
  a real company website URL, an internal capex spreadsheet with vague row labels,
  a scanned PDF where OCR quality is poor.

---

## Next steps

1. **Re-ingest corpus** — `TRUNCATE documents;` then `scripts/ingest_corpus.py --strategy naive_pdf`
2. **Build corpus search tool** — wrap `DocumentRepository.vector_search()` as an agent
   tool; run eval and compare to `llm_direct` baseline
3. **Grounded reasoning prompt/constraint** — experiment with forcing the agent to only
   make claims it can back with a retrieved document section
4. **hard_02 eval set** — realistic messy attachments (URLs, internal spreadsheets,
   incomplete or contradictory data)
