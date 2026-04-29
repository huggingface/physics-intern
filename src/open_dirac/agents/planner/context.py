"""Planner context renderer: strategy-revision input from research state."""

from __future__ import annotations

from ...rendering.shared import (
    _dedup_failed_approaches,
    _problem_guidelines,
    _render_sanity_checks,
    render_background_survey_xml,
)
from ...research_state import (
    Hypothesis,
    HypothesisStatus,
    ResearchState,
)


def _render_entity_detail(h: Hypothesis, is_er: bool = False) -> str:
    """Render enriched entity description for planner revision context."""
    summary_limit = 300 if is_er else 150
    status_tag = (
        "VERIFIED"
        if h.status == HypothesisStatus.ESTABLISHED
        else (h.review.verdict if h.review else "PENDING REVIEW")
    )
    lines = [f"{h.id}: {h.statement}, {status_tag}"]
    deps = ", ".join(h.depends_on) if h.depends_on else "none"
    lines.append(f"  depends_on: {deps}")
    if is_er and h.derivation:
        lines.append(f"  derivation (excerpt): {h.derivation[:300]}")
    for ev in h.evidence:
        summary = (ev.summary or "")[:summary_limit]
        lines.append(f"  evidence: [{ev.id}] {ev.type} — {summary}")
    if h.review:
        review_summary = (h.review.summary or "")[:summary_limit]
        lines.append(f"  review: {h.review.verdict} — {review_summary}")
    return "\n".join(lines)


def _render_rq_detail(rq) -> str:
    """Render enriched RQ description for planner revision context."""
    n_ev = len(rq.evidence)
    lines = [
        f"{rq.id}: {rq.question}, {rq.status.value.upper()}, {n_ev} evidence item{'s' if n_ev != 1 else ''}"
    ]
    for ev in rq.evidence:
        summary = (ev.summary or "")[:150]
        lines.append(f"  evidence: [{ev.id}] {ev.type} — {summary}")
    return "\n".join(lines)


def render_planner_revise_context(state: ResearchState, trigger_text: str) -> str:
    """Render context for the planner's strategy-revision mode.

    Provides: problem statement, background survey, current strategy,
    revision trigger, enriched entity descriptions, dead ends, research notes,
    and conventions.
    """
    parts: list[str] = []

    # Problem Statement
    parts.append(
        f"<problem-statement>\n{state.problem_statement or '(No problem statement.)'}\n</problem-statement>"
    )

    if state.answer_template:
        parts.append(f"<answer-template>\n{state.answer_template}\n</answer-template>")
    parts.append(_problem_guidelines())

    # Background Survey (excludes conventions and sanity checks — rendered separately)
    survey_ctx = render_background_survey_xml(state)
    if survey_ctx:
        parts.append(f"<background-survey>\n{survey_ctx}\n</background-survey>")

    # Research state — conventions, established results (enriched), dead ends
    rs_parts: list[str] = []
    if state.conventions:
        rs_parts.append(f"<conventions>\n{state.conventions}\n</conventions>")

    # Established Results only (enriched detail, not one-liner)
    er_lines: list[str] = []
    for h in sorted(state.hypotheses.values(), key=lambda h: h.id):
        if h.status == HypothesisStatus.ESTABLISHED:
            er_lines.append(_render_entity_detail(h, is_er=True))
    if er_lines:
        rs_parts.append(
            "<established-results>\n"
            + "\n\n".join(er_lines)
            + "\n</established-results>"
        )

    # Dead Ends
    de_lines: list[str] = []
    deduped = _dedup_failed_approaches(state.failed_approaches)
    for fa in deduped:
        line = f"- {fa.description}"
        if fa.reason:
            line += f" (Reason: {fa.reason})"
        de_lines.append(line)
    fa_descriptions = {fa.description for fa in deduped}
    for h in sorted(state.hypotheses.values(), key=lambda h: h.id):
        if h.status == HypothesisStatus.ABANDONED:
            desc = f"Abandoned {h.id} — {h.statement}"
            if desc not in fa_descriptions:
                de_lines.append(f"- {desc}")
    if de_lines:
        rs_parts.append("<dead-ends>\n" + "\n".join(de_lines) + "\n</dead-ends>")

    if rs_parts:
        parts.append(
            "<research-state>\n" + "\n\n".join(rs_parts) + "\n</research-state>"
        )

    # Current strategy (top-level, planner's own output being revised)
    strat = state.strategy or "(No strategy set.)"
    parts.append(f"<current-strategy>\n{strat}\n</current-strategy>")

    # Current sanity checks (editable by the planner)
    if state.sanity_checks:
        parts.append(
            _render_sanity_checks(state.sanity_checks, tag="current-sanity-checks")
        )

    # Revision Trigger
    parts.append(f"<revision-trigger>\n{trigger_text}\n</revision-trigger>")

    return "\n\n".join(parts)
