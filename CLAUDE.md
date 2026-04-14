# EU Taxonomy RAG — AI Session Instructions

## Project purpose
Experimental RAG platform for EU Taxonomy documents. Solo research project.
The developer experiments with different ingestion strategies, RAG pipelines,
and agent designs. Prioritise clean interfaces and swappability over cleverness.

## Key conventions
- Package lives in src/taxonomy_rag/
- Config is always imported from taxonomy_rag.config (settings singleton)
- DB layer: psycopg3 raw SQL, no ORM. Repository pattern.
- Ingestion strategies implement the Chunker or DocumentParser Protocol.
  Add new strategies as new files — never modify existing ones.
- RAG pipelines in rag/ all expose .ingest() and .query() with the same signature.
- Evaluation: run scripts/evaluate.py to benchmark against eval/golden_dataset.json.

## What not to do
- Do not add an ORM (SQLAlchemy etc.) — raw psycopg is intentional.
- Do not change the DB schema in 00_init.sql without flagging it — the vector
  dimension (384) is coupled to the embedding model choice.
- Do not implement ingestion parsers or chunkers without being asked — these are
  the core experiments and the developer fills them in deliberately.

## Project state
notes.md tracks the experimental plan, current status, and next steps.
Read it at the start of a session to get oriented quickly.

## Running the project
```
docker compose up -d          # start Postgres
uv run python scripts/ingest.py data/raw/2021_2139_EN.pdf
uv run python scripts/evaluate.py --questions eval/simple_v1/questions.json \
    --prompt prompts/base_v1.txt --agent llm_direct
```
