# Evaluator Guide — Hard Questions (hard_01)

## What this suite tests

Three adversarial energy questions drawn from a realistic deal-pack scenario. Each question gives the agent a mix of authoritative documents (permits, assurance extracts, contracts, quarterly reports) and non-authoritative material (marketing brochures, CEO quotes, websites). The adversarial intent is to test whether the agent can rank evidence correctly — preferring hard contractual and regulatory documents over public-facing claims — and apply EU Taxonomy classification rules precisely rather than accepting the company's own framing.

The three traps used:
- **Evidence Overload / Authority Ranking** — multiple documents, some authoritative, some not; agent must weight them correctly
- **Scope Confusion (Marketing vs Contract)** — company claims it does X; contractual evidence shows it does Y
- **NACE Spoofing** — company asserts a NACE code and activity label that do not match the asset's actual technical function

Attachments are real documents the agent must retrieve and read. Testing retrieval from company documents is part of the experiment.

---

## How to run an evaluation

1. Pick a question from `questions.json`.
2. Provide the agent with the `context` field and the documents listed in `attachments` (found in `attachments/<question-id>/`).
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

- Did the agent resist the adversarial trap, or did it accept the company's framing uncritically?
- Did it correctly rank authoritative documents (permits, contracts, assurance extracts) above marketing material?
- Did it cite the right regulatory provision or at least reason toward the correct one?
- Was the reasoning sound even if a specific citation was missed?
- Did it hedge appropriately when evidence was ambiguous, or did it over-commit to a wrong answer?

These observations are more useful for comparing agent versions than the score alone.
