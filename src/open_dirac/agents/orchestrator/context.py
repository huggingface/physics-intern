"""Orchestrator context renderer: compact one-liner view of research state."""

from __future__ import annotations

from ...rendering.shared import _dedup_failed_approaches, _render_sanity_checks
from ...research_state import (
    HypothesisStatus,
    ResearchState,
    RQStatus,
)


def render_orchestrator_slim_state(
    state: ResearchState,
    *,
    max_open_rqs: int | None = None,
) -> str:
    """Render compact one-liner research state for the orchestrator.

    Target ~8K tokens.  Keeps conventions and strategy in full; renders
    entities as one-liners (no derivations, evidence details, or review text).

    If *max_open_rqs* is given, a cap annotation is added to the
    ``<research-questions>`` section so the orchestrator knows the limit.
    """
    parts: list[str] = []

    # Conventions (full)
    conv = (
        state.conventions
        or "(To be populated by the orchestrator as conventions become clear.)"
    )
    parts.append(f"<conventions>\n{conv}\n</conventions>")

    # Strategy (full)
    strat = state.strategy or "(No strategy set.)"
    parts.append(f"<strategy>\n{strat}\n</strategy>")

    # Sanity checks
    if state.sanity_checks:
        parts.append(_render_sanity_checks(state.sanity_checks))

    # Established Results — one-liner per ER
    ers = sorted(
        [
            h
            for h in state.hypotheses.values()
            if h.status == HypothesisStatus.ESTABLISHED
        ],
        key=lambda h: h.id,
    )
    if ers:
        er_lines = [f"- {h.id}: {h.statement}, VERIFIED" for h in ers]
        parts.append(
            "<established-results>\n" + "\n".join(er_lines) + "\n</established-results>"
        )

    # Working Hypotheses — one-liner per WH
    whs = sorted(
        [h for h in state.hypotheses.values() if h.status == HypothesisStatus.WORKING],
        key=lambda h: h.id,
    )
    if whs:
        wh_lines: list[str] = []
        for h in whs:
            if h.review:
                verdict = h.review.verdict
                wh_lines.append(f"- {h.id}: {h.statement}, {verdict}")
            elif h.evidence:
                wh_lines.append(f"- {h.id}: {h.statement}, PENDING REVIEW")
            else:
                wh_lines.append(f"- {h.id}: {h.statement}, no evidence yet")
        parts.append("<hypotheses>\n" + "\n".join(wh_lines) + "\n</hypotheses>")

    # Research Questions — one-liner per RQ (omit resolved→ER, already in established-results)
    rq_lines: list[str] = []
    n_open_rqs = 0
    if state.research_questions:
        for rq in sorted(state.research_questions.values(), key=lambda r: r.id):
            if rq.status == RQStatus.OPEN:
                n_open_rqs += 1
                n_ev = len(rq.evidence)
                ev_note = (
                    f", {n_ev} evidence item{'s' if n_ev != 1 else ''}" if n_ev else ""
                )
                rq_lines.append(f"- {rq.id}: {rq.question}, OPEN{ev_note}")
            elif rq.resolved_to and all(t.startswith("ER-") for t in rq.resolved_to):
                continue  # already visible in established-results
            elif rq.resolved_to:
                rq_lines.append(
                    f"- {rq.id}: {rq.question}, RESOLVED → {', '.join(rq.resolved_to)}"
                )
            else:
                rq_lines.append(f"- {rq.id}: {rq.question}, {rq.status.upper()}")
    if max_open_rqs is not None:
        cap_note = f"Open RQ cap: {max_open_rqs} (currently {n_open_rqs} open"
        if n_open_rqs >= max_open_rqs:
            cap_note += (
                " — limit reached, resolve existing RQs before creating new ones"
            )
        cap_note += ")"
        rq_lines.insert(0, cap_note)
    if rq_lines:
        parts.append(
            "<research-questions>\n" + "\n".join(rq_lines) + "\n</research-questions>"
        )

    # Dead Ends — one-liner per entry (truncated for context budget)
    def _trunc(s: str, cap: int) -> str:
        return s if len(s) <= cap else s[: cap - 1] + "…"

    dead_lines: list[str] = []
    deduped = _dedup_failed_approaches(state.failed_approaches)
    for fa in deduped:
        desc = _trunc(fa.description, 150)
        reason = f" ({_trunc(fa.reason, 120)})" if fa.reason else ""
        dead_lines.append(f"- {desc}{reason}")
    fa_descriptions = {fa.description for fa in deduped}
    for h in sorted(state.hypotheses.values(), key=lambda h: h.id):
        if h.status == HypothesisStatus.ABANDONED:
            desc = f"Abandoned {h.id} — {h.statement}"
            if desc not in fa_descriptions:
                dead_lines.append(f"- {_trunc(desc, 150)}")
    if dead_lines:
        parts.append("<dead-ends>\n" + "\n".join(dead_lines) + "\n</dead-ends>")

    return "\n\n".join(parts)
