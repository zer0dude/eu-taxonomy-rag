# EU Taxonomy RAG — Working Notes

## Experimental plan

Build incrementally. Each step should be testable and produce a measurable result before moving on.

1. **Evaluation baseline** — refine golden questions, lock down scoring method
2. **Bare LLM baseline** — run golden questions through the LLM with no context or tools
   - Records the floor: what does the LLM already know from training?
3. **Agent tracing / run logs** — structured per-question log of what the agent did
   (tool calls, reasoning, iterations) so results can be inspected and compared
4. **PDF spike** — extract raw text from one real document in a notebook; look at it before committing to a chunking strategy
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
- [ ] PDF spike  ← next
- [ ] Naive ingestion
- [ ] Naive RAG tool
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

1. **PDF spike** — extract and inspect raw text from 2021_2139_EN.pdf in a notebook
3. **Implement PDFParser + NaiveChunker** — ingest one EU Taxonomy document
4. **Build corpus search tool** — wrap `DocumentRepository.vector_search()` as an agent
   tool; run eval and compare to `llm_direct` baseline
5. **Grounded reasoning prompt/constraint** — experiment with forcing the agent to only
   make claims it can back with a retrieved document section
6. **hard_02 eval set** — realistic messy attachments (URLs, internal spreadsheets,
   incomplete or contradictory data)
