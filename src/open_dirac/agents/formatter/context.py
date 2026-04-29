"""Formatter context renderer: focused view for producing ANSWER.md."""

from __future__ import annotations

from ...rendering.shared import _problem_guidelines, _render_sanity_checks
from ...research_state import (
    HypothesisStatus,
    ResearchState,
    RQStatus,
)


def render_formatter_context(
    state: ResearchState,
    answer_ers: list[str] | None = None,
    *,
    best_effort: bool = False,
) -> str:
    """Render focused context for the formatter agent using XML tags.

    Includes only what the formatter needs: problem statement, conventions,
    established results (with evidence/review), and a brief warning for any
    remaining open RQs or WHs.

    If *answer_ers* is provided, an ``<answer-structure>`` section is emitted
    listing the ER IDs in the order chosen by the orchestrator.
    """
    parts: list[str] = []

    # Problem Statement
    parts.append(
        f"<problem-statement>\n{state.problem_statement or '(No problem statement.)'}\n</problem-statement>"
    )

    if state.answer_template:
        parts.append(f"<answer-template>\n{state.answer_template}\n</answer-template>")
    parts.append(_problem_guidelines())

    # Research state — conventions + sanity checks
    rs_parts: list[str] = []
    if state.conventions:
        rs_parts.append(f"<conventions>\n{state.conventions}\n</conventions>")
    if state.sanity_checks:
        rs_parts.append(_render_sanity_checks(state.sanity_checks))
    if rs_parts:
        parts.append("<research-state>\n" + "\n".join(rs_parts) + "\n</research-state>")

    # Answer structure hint from orchestrator
    if answer_ers:
        er_list = "\n".join(f"- {er_id}" for er_id in answer_ers)
        parts.append(
            "<answer-structure>\n"
            "The orchestrator identified these established results as the "
            "key answers, in this order:\n"
            f"{er_list}\n"
            "</answer-structure>"
        )

    # Established Results
    ers = sorted(
        [
            h
            for h in state.hypotheses.values()
            if h.status == HypothesisStatus.ESTABLISHED
        ],
        key=lambda h: h.id,
    )
    if ers:
        er_lines: list[str] = []
        for h in ers:
            h_parts: list[str] = []
            if h.statement:
                h_parts.append(f"Statement: {h.statement}")
            if h.derivation:
                h_parts.append(f"Derivation: {h.derivation}")
            if h.evidence:
                for ev in h.evidence:
                    ev_parts: list[str] = []
                    if ev.method:
                        ev_parts.append(f"Method: {ev.method}")
                    if ev.result:
                        ev_parts.append(f"Result: {ev.result}")
                    if ev.confidence:
                        ev_parts.append(f"Confidence: {ev.confidence}")
                    h_parts.append(
                        "<evidence>\n" + "\n".join(ev_parts) + "\n</evidence>"
                    )
            if h.review:
                h_parts.append(f"Review verdict: {h.review.verdict}")
            er_lines.append(
                f'<result id="{h.id}">\n' + "\n".join(h_parts) + "\n</result>"
            )
        parts.append(
            "<established-results>\n" + "\n".join(er_lines) + "\n</established-results>"
        )
    else:
        parts.append(
            "<established-results>\n(No established results.)\n</established-results>"
        )

    # Best-effort: include working hypotheses with full evidence so the
    # formatter can attempt an answer even without full ER coverage.
    if best_effort:
        working_whs_full = sorted(
            [
                h
                for h in state.hypotheses.values()
                if h.status == HypothesisStatus.WORKING
            ],
            key=lambda h: h.id,
        )
        if working_whs_full:
            wh_lines: list[str] = []
            for h in working_whs_full:
                h_parts_w: list[str] = []
                if h.statement:
                    h_parts_w.append(f"Statement: {h.statement}")
                if h.derivation:
                    h_parts_w.append(f"Derivation: {h.derivation}")
                if h.evidence:
                    for ev in h.evidence:
                        ev_parts_w: list[str] = []
                        if ev.method:
                            ev_parts_w.append(f"Method: {ev.method}")
                        if ev.result:
                            ev_parts_w.append(f"Result: {ev.result}")
                        if ev.confidence:
                            ev_parts_w.append(f"Confidence: {ev.confidence}")
                        h_parts_w.append(
                            "<evidence>\n" + "\n".join(ev_parts_w) + "\n</evidence>"
                        )
                if h.review:
                    h_parts_w.append(f"Review verdict: {h.review.verdict}")
                wh_lines.append(
                    f'<working-hypothesis id="{h.id}">\n'
                    + "\n".join(h_parts_w)
                    + "\n</working-hypothesis>"
                )
            parts.append(
                "<unverified-results>\n"
                "The following working hypotheses have NOT been fully verified but may "
                "contain useful partial results:\n"
                + "\n".join(wh_lines)
                + "\n</unverified-results>"
            )

    # Unresolved items warning
    open_rqs = [
        rq for rq in state.research_questions.values() if rq.status == RQStatus.OPEN
    ]
    working_whs = [
        h for h in state.hypotheses.values() if h.status == HypothesisStatus.WORKING
    ]
    if open_rqs or working_whs:
        warning_lines: list[str] = []
        for rq in sorted(open_rqs, key=lambda r: r.id):
            warning_lines.append(f"- {rq.id} [OPEN]: {rq.question}")
        for h in sorted(working_whs, key=lambda h: h.id):
            warning_lines.append(f"- {h.id} [WORKING]: {h.statement}")
        parts.append(
            "<unresolved-items>\n" + "\n".join(warning_lines) + "\n</unresolved-items>"
        )

    return "\n\n".join(parts)
