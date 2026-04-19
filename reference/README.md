# Reference Code

This directory contains prior-project RAG pipeline code kept as implementation reference.

## `rag/`

`naive.py`, `hybrid.py`, and `advanced.py` are end-to-end retrieve-then-generate pipelines
from an earlier project. They are **not imported by any active code** and have no tests.

Use them as reference when implementing the proper retrieval primitives in
`src/taxonomy_rag/retrieval/`:

| Reference file | Implement as |
|---|---|
| `rag/hybrid.py` — RRF fusion logic | `retrieval/hybrid.py` — `HybridRetrieval(Retriever)` |
| `rag/advanced.py` — HyDE + cross-encoder reranking | `retrieval/advanced.py` — `AdvancedRetrieval(Retriever)` |

The key extraction: pull the retrieval logic (embed, search, rerank) out of these monoliths
and into classes that implement the `Retriever` protocol in `retrieval/base.py`. The
generate step belongs to the agent, not the retriever.
