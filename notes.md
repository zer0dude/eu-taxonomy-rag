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

- [ ] Evaluation baseline
- [ ] Bare LLM baseline
- [ ] PDF spike
- [ ] Naive ingestion
- [ ] Naive RAG
- [ ] Retrieval iteration
- [ ] Agent layer

---

## Learnings

_Updated as experiments run._

---

## Open questions

_Things to investigate or decide._

---

## Next steps

- Refine golden questions and evaluation scoring
- Run bare LLM baseline on current golden questions
