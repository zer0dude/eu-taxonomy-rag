# EU Taxonomy RAG — Working Notes

## Experimental plan

Build incrementally. Each step should be testable and produce a measurable result before moving on.

1. **Evaluation baseline** — refine golden questions, lock down scoring method
2. **Bare LLM baseline** — run golden questions through the LLM with no context or tools
   - Records the floor: what does the LLM already know from training?
3. **PDF spike** — extract raw text from one real document in a notebook; look at it before committing to a chunking strategy
4. **Naive ingestion** — implement `PDFParser` + `NaiveChunker`, ingest one document
5. **Naive RAG** — `NaiveRAG` (vector search, already implemented); run evaluation and compare to bare LLM floor
6. **Iterate on retrieval** — vary chunker, retrieval method (`HybridRAG`, `AdvancedRAG`), compare scores
7. **Agent layer** — wrap reliable retrieval in a LangChain tool; add reasoning loop
   - Note: only add the agent once the tools it calls are trustworthy

**Key principle:** the agent layer only adds value if the underlying tools work. Don't add reasoning complexity on top of broken retrieval.

---

## Current status

- [x] Evaluation baseline
- [ ] Bare LLM baseline  ← in progress
- [ ] PDF spike
- [ ] Naive ingestion
- [ ] Naive RAG
- [ ] Retrieval iteration
- [ ] Agent layer

---

## Eval infrastructure (completed)

**scripts/evaluate.py** — modular eval script. Three interchangeable axes:
- `--questions` — any questions JSON (simple_v1, golden_dataset_v1, hard_01, ...)
- `--prompt` — any prompt file from `prompts/`
- `--agent` — any registered agent name (mock, llm_direct, ...)

Each run saves to `runs/{YYYYMMDD}_{HHMMSS}_{agent}_{question_set}/`:
- `metadata.json` — machine-readable run summary
- `outcomes.csv` — one row per question; human fills `human_score` and `human_notes`

**Question sets (no attachments — usable now):**
- `eval/simple_v1/questions.json` — 10 manufacturing eligibility questions, adversarial
- `eval/golden_dataset_v1/questions.json` — 3 baseline wind/solar/safeguards questions (converted from `golden_dataset.json`)

**Question sets (attachments required — not usable yet):**
- `eval/hard_01/questions.json` — 3 adversarial energy questions with PDF attachments

**Agents:**
- `mock` — hardcoded fixed string; validates pipeline plumbing
- `llm_direct` — bare Anthropic API call (Claude Haiku 4.5), system prompt from file, no retrieval

**Prompts:**
- `prompts/base_v1.txt` — EU Taxonomy expert, eligibility vs alignment framing

---

## Learnings

_Updated as experiments run._

---

## Open questions

_Things to investigate or decide._

---

## Next steps

- Run bare LLM baseline (llm_direct + base_v1) on simple_v1 and golden_dataset_v1; human-score the outcomes CSV
- PDF spike in a notebook: extract and inspect raw text from 2021_2139_EN.pdf
