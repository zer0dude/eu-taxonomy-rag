# Evaluator Guide — Simple Questions (simple_v1)

## What this suite tests

Ten eligibility mapping questions covering the Manufacturing section of the EU Taxonomy Climate Delegated Act (Annex I). Each question describes a facility's actual operation alongside a distracting element — a marketing claim, a contradictory website statement, a sustainability headline, or missing information. The adversarial intent is to test whether the agent maps the activity to the correct Taxonomy reference based on technical dossier evidence, ignoring noise.

No attachments are used. The agent works only from the text in the `context` field.

The adversarial types used:
- **Misdirection Data** — irrelevant sustainability claims or misleading headlines present in the context
- **Contradictory Data** — the dossier and a public statement describe different activities
- **Insufficient Data** — not enough information to uniquely resolve the mapping
- **Misinformation** — a non-existent Taxonomy activity reference is cited as authoritative

---

## How to run an evaluation

1. Pick a question from `questions.json`.
2. Provide the agent with the `context` field only (no attachments).
3. Ask the agent the `question` exactly as written.
4. Collect the agent's free-text response.
5. Score using the rubric below, guided by `evaluator_notes` in `questions.json`.

For each question, `evaluator_notes` contains:
- `what_to_look_for` — the key points a good answer must cover
- `key_citations` — the regulatory references that should appear (or be implicitly correct)
- `reference_answer` — a full model answer for comparison

---

## Scoring rubric

| Score | Meaning |
|-------|---------|
| 4 | Covers all key points in `what_to_look_for`, cites correct regulatory sections, no material errors |
| 3 | Covers most key points, minor gaps or imprecise citations |
| 2 | Partial — gets the gist but misses important distinctions or cites wrong sections |
| 1 | Significant errors, missed the core point, or hallucinated regulatory content |

---

## What to note beyond the score

Record qualitative observations alongside the numeric score:

- Did the agent resist the adversarial trap (marketing claim, contradictory statement, non-existent activity reference)?
- Did it correctly prioritise dossier and NACE evidence over branding language?
- Did it correctly flag insufficient data rather than forcing an answer (relevant for MAN-EASY-3.1-004)?
- Did it identify a hallucinated activity reference as non-existent rather than accepting it (relevant for MAN-EASY-3.22-010)?
- Was the Taxonomy section cited correctly, and was the distinction between similar activities handled well (e.g. 3.10 vs 3.2, 3.18 vs 3.3)?
