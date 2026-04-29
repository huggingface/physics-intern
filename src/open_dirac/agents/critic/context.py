"""Critic context renderer: strategic review view of research state."""

from __future__ import annotations

from ...rendering.shared import (
    _render_sanity_checks,
    render_background_survey_xml,
    render_research_context_xml,
)
from ...research_state import (
    CritiqueStatus,
    Hypothesis,
    HypothesisStatus,
    ResearchState,
    Severity,
)


def render_critic_previous_critiques(state: ResearchState) -> str:
    """Render previous critiques for the deep critic context.

    Unlike the orchestrator version, includes RESOLVED critiques so the
    critic can see what was already filed and avoid re-filing.
    """
    parts: list[str] = []

    # Active critiques first
    active = [c for c in state.critiques.values() if c.status == CritiqueStatus.ACTIVE]
    for c in sorted(active, key=lambda c: c.id):
        target_str = ", ".join(c.targets) if c.targets else "general"
        blocking = "true" if c.severity == Severity.HIGH else "false"
        content = c.argument or ""
        parts.append(
            f'<critique id="{c.id}" severity="{c.severity}" blocking="{blocking}"'
            f' status="UNRESOLVED" target="{target_str}">\n{content}\n</critique>'
        )

    # Resolved/withdrawn critiques — concise, with resolution
    resolved = [
        c for c in state.critiques.values() if c.status != CritiqueStatus.ACTIVE
    ]
    for c in sorted(resolved, key=lambda c: c.id):
        target_str = ", ".join(c.targets) if c.targets else "general"
        res_type = c.resolution_type or "unknown"
        resolution = c.resolution or ""
        content = c.argument or ""
        parts.append(
            f'<critique id="{c.id}" severity="{c.severity}"'
            f' status="RESOLVED" resolution-type="{res_type}" target="{target_str}">'
            f"\n{content}"
            + (f"\nResolution: {resolution}" if resolution else "")
            + "\n</critique>"
        )

    # Clean reviews
    if state.critic_clean_reviews:
        review_lines: list[str] = []
        for rev in sorted(
            state.critic_clean_reviews, key=lambda r: r.get("iteration", 0)
        ):
            review_lines.append(
                f"Iteration {rev.get('iteration', '?')}: {rev.get('summary', '')}"
            )
        parts.append(
            "<clean-reviews>\n" + "\n".join(review_lines) + "\n</clean-reviews>"
        )

    return "\n".join(parts)


def render_critic_context(state: ResearchState, iteration: int) -> str:
    """Render strategic context for the deep critic using XML tags.

    Provides: problem statement, strategy, conventions, RQ list,
    hypothesis summaries (evidence/review one-liners), background survey,
    and previous critiques.  No derivations, scripts, reasoning, or approach text.
    """
    parts: list[str] = []

    # 1. Research context — problem statement + answer template
    parts.append(render_research_context_xml(state))

    # 2. Background survey
    survey_ctx = render_background_survey_xml(state)
    if survey_ctx:
        parts.append(f"<background-survey>\n{survey_ctx}\n</background-survey>")

    # 3. Research state — strategy, conventions, sanity checks, entities
    rs_parts: list[str] = []

    conv = state.conventions or "(No conventions set.)"
    rs_parts.append(f"<conventions>\n{conv}\n</conventions>")

    strat = state.strategy or "(No strategy set.)"
    rs_parts.append(f"<strategy>\n{strat}\n</strategy>")

    if state.sanity_checks:
        rs_parts.append(_render_sanity_checks(state.sanity_checks))

    # Research Questions
    if state.research_questions:
        rq_lines: list[str] = []
        for rq in sorted(state.research_questions.values(), key=lambda r: r.id):
            rq_lines.append(
                f'<rq id="{rq.id}" status="{rq.status.upper()}">{rq.question}</rq>'
            )
        rs_parts.append(
            "<research-questions>\n" + "\n".join(rq_lines) + "\n</research-questions>"
        )

    # Compute last critic iteration from clean reviews and filed critiques
    last_critic_iter = 0
    if state.critic_clean_reviews:
        last_critic_iter = max(
            r.get("iteration", 0) for r in state.critic_clean_reviews
        )
    for c in state.critiques.values():
        if c.iteration_filed > last_critic_iter:
            last_critic_iter = c.iteration_filed

    # Collect hypothesis IDs targeted by critiques (used for survived-critic check)
    critic_targets_since: dict[
        str, int
    ] = {}  # hid -> latest critique iteration targeting it
    for c in state.critiques.values():
        for t in c.targets:
            if c.iteration_filed > critic_targets_since.get(t, 0):
                critic_targets_since[t] = c.iteration_filed

    def _critic_hyp_parts(h: Hypothesis) -> list[str]:
        h_parts: list[str] = []
        if h.statement:
            h_parts.append(f"Statement: {h.statement}")
        if h.depends_on:
            h_parts.append(f"Depends on: {', '.join(h.depends_on)}")
        if h.evidence:
            for ev in h.evidence:
                result_short = (
                    (ev.result[:300] + "...")
                    if ev.result and len(ev.result) > 300
                    else (ev.result or "")
                )
                h_parts.append(
                    f"Evidence ({ev.type}): {ev.method or 'not specified'}, confidence={ev.confidence or '?'}, Result: {result_short}"
                )
        if h.review:
            v = h.review
            h_parts.append(f"Review: {v.verdict} — {v.summary}")
            survived_critic = (
                h.iteration_modified <= last_critic_iter
                and critic_targets_since.get(h.id, 0) < h.iteration_modified
            )
            if not survived_critic and v.details:
                details_truncated = (
                    (v.details[:1000] + "...") if len(v.details) > 1000 else v.details
                )
                h_parts.append(f"Review details: {details_truncated}")
        return h_parts

    # Hypotheses (working)
    whs = sorted(
        [h for h in state.hypotheses.values() if h.status == HypothesisStatus.WORKING],
        key=lambda h: h.id,
    )
    if whs:
        hyp_lines: list[str] = []
        for h in whs:
            hyp_lines.append(
                f'<hypothesis id="{h.id}">\n'
                + "\n".join(_critic_hyp_parts(h))
                + "\n</hypothesis>"
            )
        rs_parts.append("<hypotheses>\n" + "\n".join(hyp_lines) + "\n</hypotheses>")

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
            er_lines.append(
                f'<result id="{h.id}">\n'
                + "\n".join(_critic_hyp_parts(h))
                + "\n</result>"
            )
        rs_parts.append(
            "<established-results>\n" + "\n".join(er_lines) + "\n</established-results>"
        )

    parts.append("<research-state>\n" + "\n\n".join(rs_parts) + "\n</research-state>")

    # 4. Previous Critiques (critic needs both active AND resolved to avoid re-filing)
    critique_xml = render_critic_previous_critiques(state)
    parts.append(f"<previous-critiques>\n{critique_xml}\n</previous-critiques>")

    return "\n\n".join(parts)
