"""HTML report generator for eval run results.

Produces a self-contained report.html from run data. No external dependencies
at render time — marked.js is embedded inline so the file works offline.

Public API:
    build_report_html(run_id, metadata, rows, traces) -> str
"""

from __future__ import annotations

import html
import json
from pathlib import Path

_VENDOR_DIR = Path(__file__).parent / "_vendor"

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 14px; line-height: 1.5; background: #f0f2f5; color: #333;
}
.container { max-width: 1060px; margin: 0 auto; padding: 24px 20px; }

/* ── Run header ── */
.run-header {
    background: #1e2235; color: #e8eaf0; padding: 20px 24px;
    border-radius: 10px; margin-bottom: 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.run-header h1 { font-size: 16px; font-weight: 700; margin-bottom: 10px; letter-spacing: 0.02em; }
.run-meta { display: flex; flex-wrap: wrap; gap: 6px 20px; font-size: 12px; color: #9ba3bb; margin-bottom: 14px; }
.run-meta span b { color: #c8cde0; font-weight: 600; }
.comments-label { font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #7a82a0; margin-bottom: 5px; }
#run-comments {
    width: 100%; min-height: 54px;
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.15);
    border-radius: 5px; color: #e0e4f0; padding: 8px 10px;
    font-size: 13px; resize: vertical; font-family: inherit;
}
#run-comments::placeholder { color: rgba(255,255,255,0.25); }
#run-comments:focus { outline: none; border-color: rgba(100,140,220,0.6); }

/* ── Question card ── */
.question-card {
    background: white; border-radius: 10px; margin-bottom: 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.09); overflow: hidden;
}
.card-header {
    background: #2a2d3e; color: white;
    padding: 11px 16px; display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
}
.q-id { font-weight: 700; font-size: 13px; font-family: 'SFMono-Regular', Consolas, monospace; }
.card-stats { font-size: 11px; color: #888fa8; margin-left: auto; }
.badges { display: flex; gap: 4px; flex-wrap: wrap; margin-left: 8px; }
.badge { font-size: 10px; padding: 2px 7px; border-radius: 3px; font-weight: 600; letter-spacing: 0.02em; }
.badge-simple  { background: #1a7a42; color: #d4f7e5; }
.badge-hard    { background: #8b1a1a; color: #ffd4d4; }
.badge-adv     { background: #5a2080; color: #e8d4ff; }
.badge-tag     { background: #1a4a7a; color: #d4e8ff; }
.card-body { padding: 18px 18px 14px; }

/* ── Fields ── */
.field { margin-bottom: 14px; }
.field:last-of-type { margin-bottom: 0; }
.field-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.09em;
    color: #8a93a8; text-transform: uppercase; margin-bottom: 5px;
}
.field-body { font-size: 13px; }
.field-context .field-body {
    background: #f6f7f9; border-left: 3px solid #b0b8c8;
    padding: 8px 12px; border-radius: 0 5px 5px 0;
}
.field-question .field-body { font-weight: 600; font-size: 14px; color: #111; padding: 4px 0; }
.field-reference .field-body {
    background: #eef5fd; border-left: 3px solid #4a90d9;
    padding: 8px 12px; border-radius: 0 5px 5px 0;
}
.field-wtlf .field-body, .field-citations .field-body {
    background: #fdfaf3; border-left: 3px solid #d4a800;
    padding: 8px 12px; border-radius: 0 5px 5px 0;
}
.field-body ul { margin-left: 18px; }
.field-body li { margin-bottom: 3px; }
.field-attachments .field-body { font-size: 12px; color: #666; font-style: italic; }
.field-answer .field-body {
    background: #f1faf3; border-left: 3px solid #2ea04f;
    padding: 10px 14px; border-radius: 0 5px 5px 0;
}

/* ── Markdown rendering ── */
.md-body h1, .md-body h2, .md-body h3, .md-body h4 { margin: 10px 0 5px; line-height: 1.3; }
.md-body h1 { font-size: 15px; }
.md-body h2 { font-size: 14px; border-bottom: 1px solid #d8dde5; padding-bottom: 3px; }
.md-body h3 { font-size: 13px; }
.md-body h4 { font-size: 12px; color: #444; }
.md-body p  { margin-bottom: 7px; }
.md-body ul, .md-body ol { margin: 6px 0 6px 20px; }
.md-body li { margin-bottom: 2px; }
.md-body blockquote {
    border-left: 3px solid #cdd3de; margin: 7px 0;
    padding: 4px 0 4px 12px; color: #555; background: #f8f8fa;
    border-radius: 0 3px 3px 0;
}
.md-body code { background: #eef0f4; padding: 1px 5px; border-radius: 3px; font-size: 12px; font-family: monospace; }
.md-body pre  { background: #f4f5f8; padding: 10px; border-radius: 4px; overflow-x: auto; margin: 6px 0; }
.md-body pre code { background: none; padding: 0; }
.md-body strong { font-weight: 700; }
.md-body em { font-style: italic; }
.md-body hr { border: none; border-top: 1px solid #e0e4ea; margin: 10px 0; }
.md-body table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 12px; }
.md-body th, .md-body td { border: 1px solid #d0d5e0; padding: 5px 10px; text-align: left; }
.md-body th { background: #f0f2f6; font-weight: 700; }

/* ── Reasoning collapsible ── */
details.reasoning {
    margin-top: 16px; border: 1px solid #e2e6ee; border-radius: 7px; overflow: hidden;
}
details.reasoning summary {
    padding: 9px 14px; cursor: pointer; font-size: 12px; font-weight: 600;
    color: #5a6276; background: #f5f6fa; user-select: none; list-style: none;
    display: flex; align-items: center; gap: 6px;
}
details.reasoning summary::-webkit-details-marker { display: none; }
details.reasoning summary::before {
    content: '▶'; font-size: 9px; display: inline-block;
    transition: transform 0.15s; color: #8a93a8;
}
details[open].reasoning summary::before { transform: rotate(90deg); }
details.reasoning summary:hover { background: #eceff5; }
.reasoning-body { padding: 14px 16px; border-top: 1px solid #e2e6ee; background: #fafbfd; }
.iteration { margin-bottom: 18px; padding-bottom: 18px; border-bottom: 1px solid #ebebf0; }
.iteration:last-child { margin-bottom: 0; padding-bottom: 0; border-bottom: none; }
.iteration-header {
    font-size: 10px; font-weight: 700; color: #6a7390;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 8px;
}
.iteration-reasoning { font-size: 12px; margin-bottom: 10px; color: #3a3f50; }

/* tool call */
.tool-call {
    background: white; border: 1px solid #dde2ec;
    border-radius: 5px; padding: 8px 10px; margin-top: 8px;
}
.tool-call-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.tool-name { font-family: monospace; font-size: 12px; font-weight: 700; color: #6b3fa0; }
.tool-duration { font-size: 10px; color: #999; }
pre.tool-input {
    font-size: 11px; background: #f2f4f8; padding: 6px 8px; border-radius: 3px;
    overflow-x: auto; white-space: pre-wrap; max-height: 140px; overflow-y: auto;
    color: #333; margin-bottom: 5px; border: 1px solid #e2e6ee;
}
pre.tool-result {
    font-size: 11px; background: #fffbee; padding: 6px 8px; border-radius: 3px;
    overflow-x: auto; white-space: pre-wrap; max-height: 280px; overflow-y: auto;
    border: 1px solid #ffe9a0; color: #3a3000;
}
.tool-result-meta { font-size: 11px; color: #999; font-style: italic; }

/* ── Evaluator section ── */
.evaluator-section {
    background: #f7f8fc; border: 1px solid #e0e4ee; border-radius: 7px;
    padding: 12px 14px; margin-top: 16px;
    display: grid; grid-template-columns: 70px 1fr; gap: 7px 10px; align-items: start;
}
.evaluator-section .field-label { padding-top: 7px; }
.score-input, .notes-input {
    width: 100%; padding: 6px 9px; border: 1px solid #cdd2de;
    border-radius: 4px; font-size: 13px; font-family: inherit;
    background: white; color: #333;
}
.score-input:focus, .notes-input:focus {
    outline: none; border-color: #5a90d0;
    box-shadow: 0 0 0 2px rgba(90,144,208,0.15);
}
.notes-input { resize: vertical; }

/* ── Export bar ── */
.export-bar { text-align: center; margin-top: 28px; padding-bottom: 40px; }
#export-btn {
    background: #2562b8; color: white; border: none;
    padding: 12px 32px; border-radius: 7px; font-size: 14px;
    cursor: pointer; font-weight: 600; letter-spacing: 0.02em;
    box-shadow: 0 2px 6px rgba(37,98,184,0.3);
}
#export-btn:hover { background: #1e4e96; }
"""


def _build_question_card(row: dict, trace: dict | None, eval_data: dict) -> str:
    qid = row["question_id"]

    difficulty = row.get("difficulty", "")
    adv_type   = row.get("adversarial_type", "")
    tags_raw   = row.get("tags", "")

    # Stats line from trace or row
    if trace:
        tok = trace.get("token_totals", {})
        in_tok  = tok.get("input_tokens", row.get("input_tokens", 0))
        out_tok = tok.get("output_tokens", row.get("output_tokens", 0))
        n_iter  = trace.get("total_iterations", 0)
        dur     = trace.get("duration_seconds", 0)
        stats = f"{in_tok:,} in / {out_tok:,} out · {n_iter} iter · {dur:.1f}s"
    else:
        in_tok  = row.get("input_tokens", 0)
        out_tok = row.get("output_tokens", 0)
        stats = f"{in_tok:,} in / {out_tok:,} out"

    # Badges
    badges = ""
    if difficulty:
        cls = f"badge-{difficulty}" if difficulty in ("simple", "hard") else "badge-tag"
        badges += f'<span class="badge {cls}">{html.escape(difficulty)}</span>'
    if adv_type:
        badges += f'<span class="badge badge-adv">{html.escape(adv_type)}</span>'
    for tag in tags_raw.split(";"):
        tag = tag.strip()
        if tag:
            badges += f'<span class="badge badge-tag">{html.escape(tag)}</span>'

    # Field builder
    def field(cls: str, label: str, body_html: str) -> str:
        return (
            f'<div class="field {cls}">'
            f'<div class="field-label">{label}</div>'
            f'<div class="field-body">{body_html}</div>'
            f'</div>'
        )

    context  = field("field-context",   "Context",        f'<p>{html.escape(row.get("context", ""))}</p>')
    question = field("field-question",  "Question",       f'<p>{html.escape(row.get("question", ""))}</p>')
    ref      = field("field-reference", "Reference Answer", f'<p>{html.escape(row.get("reference_answer", "—"))}</p>')

    wtlf_items = [x.strip() for x in row.get("what_to_look_for", "").split("|") if x.strip()]
    wtlf_body  = ("<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in wtlf_items) + "</ul>") if wtlf_items else "<p>—</p>"
    wtlf       = field("field-wtlf",  "What to Look For", wtlf_body)

    cit_items  = [x.strip() for x in row.get("key_citations", "").split("|") if x.strip()]
    cit_body   = ("<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in cit_items) + "</ul>") if cit_items else "<p>—</p>"
    citations  = field("field-citations", "Key Citations", cit_body)

    attachments_str = row.get("attachments", "")
    attach_html = (
        field("field-attachments", "Attachments", f"<p>{html.escape(attachments_str)}</p>")
        if attachments_str else ""
    )

    # Agent answer — rendered via marked.js from EVAL_DATA
    answer_div = field(
        "field-answer", "Agent Answer",
        f'<div class="md-body" id="ans-{html.escape(qid)}"></div>',
    )

    # Reasoning section
    reasoning_html = _build_reasoning_section(qid, trace)

    # Evaluator inputs
    evaluator = (
        f'<div class="evaluator-section">'
        f'<div class="field-label">Score</div>'
        f'<input type="text" id="score-{html.escape(qid)}" class="score-input" '
        f'data-qid="{html.escape(qid)}" placeholder="e.g. correct / partial / wrong">'
        f'<div class="field-label">Notes</div>'
        f'<textarea id="notes-{html.escape(qid)}" class="notes-input" '
        f'data-qid="{html.escape(qid)}" rows="3" placeholder="Evaluator notes…"></textarea>'
        f'</div>'
    )

    return (
        f'<div class="question-card" id="card-{html.escape(qid)}">'
        f'<div class="card-header">'
        f'<span class="q-id">{html.escape(qid)}</span>'
        f'<span class="card-stats">{html.escape(stats)}</span>'
        f'<div class="badges">{badges}</div>'
        f'</div>'
        f'<div class="card-body">'
        f'{context}{question}{attach_html}{ref}{wtlf}{citations}{answer_div}'
        f'{reasoning_html}'
        f'{evaluator}'
        f'</div>'
        f'</div>'
    )


def _build_reasoning_section(qid: str, trace: dict | None) -> str:
    if not trace:
        return (
            '<details class="reasoning">'
            '<summary>No trace recorded — rerun with <code>--log-level=full</code></summary>'
            '</details>'
        )

    iterations = trace.get("iterations", [])
    if not iterations:
        return (
            '<details class="reasoning">'
            '<summary>No iterations recorded in trace</summary>'
            '</details>'
        )

    n = trace.get("total_iterations", len(iterations))
    label = f"Show reasoning ({n} iteration{'s' if n != 1 else ''})"

    iter_parts: list[str] = []
    for it in iterations:
        idx   = it.get("index", "?")
        usage = it.get("usage", {})
        in_t  = usage.get("input_tokens", 0)
        out_t = usage.get("output_tokens", 0)

        # Reasoning text — rendered markdown via JS (id: rsn-{qid}-{idx})
        rsn_id = f"rsn-{html.escape(qid)}-{idx}"
        rsn_body = f'<div class="md-body iteration-reasoning" id="{rsn_id}"></div>'

        tool_parts: list[str] = []
        for tc in it.get("tool_calls", []):
            tool_name   = html.escape(tc.get("tool", ""))
            tool_input  = html.escape(json.dumps(tc.get("input", {}), indent=2))
            dur_ms      = tc.get("duration_ms", 0)

            if "result" in tc:
                result_block = f'<pre class="tool-result">{html.escape(tc["result"])}</pre>'
            elif "result_preview" in tc:
                preview = html.escape(tc["result_preview"])
                suffix  = "…[truncated]" if tc.get("result_truncated") else ""
                result_block = f'<pre class="tool-result">{preview}{suffix}</pre>'
            else:
                chars = tc.get("result_chars", 0)
                result_block = (
                    f'<p class="tool-result-meta">'
                    f'{chars:,} chars (rerun with <code>--log-level=full</code> to store result)'
                    f'</p>'
                )

            tool_parts.append(
                f'<div class="tool-call">'
                f'<div class="tool-call-header">'
                f'<span class="tool-name">{tool_name}</span>'
                f'<span class="tool-duration">{dur_ms:.0f} ms</span>'
                f'</div>'
                f'<pre class="tool-input">{tool_input}</pre>'
                f'{result_block}'
                f'</div>'
            )

        iter_parts.append(
            f'<div class="iteration">'
            f'<div class="iteration-header">Iteration {idx} — {in_t:,} in / {out_t:,} out tokens</div>'
            f'{rsn_body}'
            f'{"".join(tool_parts)}'
            f'</div>'
        )

    return (
        f'<details class="reasoning">'
        f'<summary>{label}</summary>'
        f'<div class="reasoning-body">{"".join(iter_parts)}</div>'
        f'</details>'
    )


def _build_eval_data_js(run_id: str, rows: list[dict], traces: dict[str, dict]) -> str:
    """Build the EVAL_DATA JS object that the browser uses to render markdown."""
    questions: dict[str, dict] = {}
    for row in rows:
        qid   = row["question_id"]
        trace = traces.get(qid)
        entry: dict = {"answer_md": row.get("agent_answer", "")}
        if trace:
            entry["reasoning"] = [
                {"index": it.get("index"), "md": it.get("reasoning", "")}
                for it in trace.get("iterations", [])
            ]
        questions[qid] = entry
    data = {"run_id": run_id, "questions": questions}
    return json.dumps(data, ensure_ascii=False)


def build_report_html(
    run_id: str,
    metadata: dict,
    rows: list[dict],
    traces: dict[str, dict],
) -> str:
    """Return a complete self-contained HTML report as a string.

    Args:
        run_id:   The run identifier (used as localStorage namespace).
        metadata: The run's metadata.json dict.
        rows:     Per-question dicts as built by evaluate.py (same fields as outcomes.csv).
        traces:   Mapping of question_id -> trace dict. May be empty.
    """
    marked_js = (_VENDOR_DIR / "marked.min.js").read_text(encoding="utf-8")
    eval_data_js = _build_eval_data_js(run_id, rows, traces)

    # Run header meta fields
    agent        = html.escape(metadata.get("agent", ""))
    qfile        = html.escape(metadata.get("questions_file", ""))
    prompt_file  = html.escape(metadata.get("prompt_file") or "(agent default)")
    timestamp    = html.escape(metadata.get("timestamp", ""))
    duration     = metadata.get("duration_seconds", 0)
    tok          = metadata.get("token_totals", {})
    in_tok       = tok.get("input_tokens", 0)
    out_tok      = tok.get("output_tokens", 0)
    n_questions  = metadata.get("total_questions", len(rows))

    run_meta = (
        f'<div class="run-meta">'
        f'<span><b>Agent:</b> {agent}</span>'
        f'<span><b>Questions:</b> {qfile} ({n_questions})</span>'
        f'<span><b>Prompt:</b> {prompt_file}</span>'
        f'<span><b>Timestamp:</b> {timestamp}</span>'
        f'<span><b>Duration:</b> {duration:.1f}s</span>'
        f'<span><b>Tokens:</b> {in_tok:,} in / {out_tok:,} out</span>'
        f'</div>'
    )

    # Build question cards
    eval_data_for_cards: dict = {}  # unused by _build_question_card but kept for signature
    cards_html = "\n".join(
        _build_question_card(row, traces.get(row["question_id"]), eval_data_for_cards)
        for row in rows
    )

    run_id_js = json.dumps(run_id)

    inline_js = f"""
const EVAL_DATA = {eval_data_js};
const RUN_ID = {run_id_js};

// ── Markdown rendering ──────────────────────────────────────────────
marked.setOptions({{ breaks: true, gfm: true }});

function renderMd(el, md) {{
    if (!md) {{ el.innerHTML = '<em style="color:#999">(empty)</em>'; return; }}
    el.innerHTML = marked.parse(md);
}}

Object.entries(EVAL_DATA.questions).forEach(([qid, data]) => {{
    const ansEl = document.getElementById('ans-' + qid);
    if (ansEl) renderMd(ansEl, data.answer_md);

    (data.reasoning || []).forEach(it => {{
        const rsnEl = document.getElementById('rsn-' + qid + '-' + it.index);
        if (rsnEl) renderMd(rsnEl, it.md);
    }});
}});

// ── localStorage persistence ────────────────────────────────────────
function lsKey(parts) {{ return ['eval', RUN_ID, ...parts].join('::'); }}

const commentsEl = document.getElementById('run-comments');
commentsEl.value = localStorage.getItem(lsKey(['comments'])) || '';
commentsEl.addEventListener('input', () => {{
    localStorage.setItem(lsKey(['comments']), commentsEl.value);
}});

document.querySelectorAll('.score-input, .notes-input').forEach(el => {{
    const qid  = el.dataset.qid;
    const type = el.classList.contains('score-input') ? 'score' : 'notes';
    el.value   = localStorage.getItem(lsKey([qid, type])) || '';
    el.addEventListener('input', () => {{
        localStorage.setItem(lsKey([qid, type]), el.value);
    }});
}});

// ── Export ─────────────────────────────────────────────────────────
document.getElementById('export-btn').addEventListener('click', () => {{
    const out = {{
        run_id: RUN_ID,
        exported_at: new Date().toISOString(),
        comments: commentsEl.value,
        questions: []
    }};
    document.querySelectorAll('.question-card').forEach(card => {{
        const qid = card.querySelector('.q-id').textContent;
        out.questions.push({{
            question_id: qid,
            score: (card.querySelector('.score-input') || {{}}).value || '',
            notes: (card.querySelector('.notes-input') || {{}}).value || ''
        }});
    }});
    const blob = new Blob([JSON.stringify(out, null, 2)], {{type: 'application/json'}});
    const a    = document.createElement('a');
    a.href     = URL.createObjectURL(blob);
    a.download = RUN_ID + '_evaluator_notes.json';
    a.click();
}});
"""

    title = html.escape(run_id)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eval: {title}</title>
<script>{marked_js}</script>
<style>{_CSS}</style>
</head>
<body>
<div class="container">

<header class="run-header">
<h1>Eval run: {title}</h1>
{run_meta}
<div class="comments-label">Evaluator comments</div>
<textarea id="run-comments" rows="2" placeholder="Overall notes about this run…"></textarea>
</header>

<div class="questions">
{cards_html}
</div>

<div class="export-bar">
<button id="export-btn">Export evaluator notes (JSON)</button>
</div>

</div>
<script>{inline_js}</script>
</body>
</html>"""
