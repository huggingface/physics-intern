"""Snapshot renderers: produce full Markdown files from ResearchState.

These are write-only — rendered once per iteration for git history and
verify.py. Never read back by agents.
"""

from __future__ import annotations

from ..utils.markdown import render_frontmatter
from ..research_state import (
    CritiqueStatus,
    HypothesisStatus,
    ResearchState,
    RQStatus,
)
from .shared import _dedup_failed_approaches


def render_background_survey(state: ResearchState) -> str:
    """Render background survey as a Markdown section (for git snapshots)."""
    has_content = (
        state.survey_background
        or state.key_insights
        or state.survey_methods
        or state.known_pitfalls
        or state.expected_answer_structure
    )
    if not has_content:
        return "(No background survey.)"

    parts: list[str] = ["# Background Survey\n"]
    if state.survey_background:
        parts.append(f"### Background\n\n{state.survey_background}\n")
    if state.key_insights:
        parts.append(f"### Key Insights\n\n{state.key_insights}\n")
    if state.survey_methods:
        parts.append(f"### Known Methods and Techniques\n\n{state.survey_methods}\n")
    if state.known_pitfalls:
        parts.append(f"### Known Pitfalls\n\n{state.known_pitfalls}\n")
    if state.expected_answer_structure:
        parts.append(
            f"### Expected Answer Structure\n\n{state.expected_answer_structure}\n"
        )
    if state.conventions:
        parts.append(f"### Conventions and Definitions\n\n{state.conventions}\n")
    if state.sanity_checks:
        lines = []
        for sc in state.sanity_checks:
            line = f"- **[{sc.id}]** {sc.predicate}"
            if sc.rationale:
                line += f" — *{sc.rationale}*"
            lines.append(line)
        parts.append("### Sanity Checks\n\n" + "\n".join(lines) + "\n")
    return "\n".join(parts)


def _research_state_body(state: ResearchState) -> str:
    """Build the body text for a research state snapshot rendering."""
    parts: list[str] = []

    # Problem Statement
    parts.append("# Problem Statement\n")
    parts.append(state.problem_statement or "(No problem statement.)")
    parts.append("")

    if state.answer_template:
        parts.append("# Expected Answer Format\n")
        parts.append(state.answer_template)
        parts.append("")

    # Conventions
    parts.append("# Conventions\n")
    parts.append(
        state.conventions
        or "(To be populated by the orchestrator as conventions become clear.)"
    )
    parts.append("")

    # Strategy
    parts.append("# Strategy\n")
    parts.append(state.strategy or "(No strategy set.)")
    parts.append("")

    # Research Questions
    open_rqs = [
        rq for rq in state.research_questions.values() if rq.status == RQStatus.OPEN
    ]
    resolved_rqs = [
        rq for rq in state.research_questions.values() if rq.status != RQStatus.OPEN
    ]
    if state.research_questions:
        parts.append("# Research Questions\n")
        for rq in sorted(open_rqs, key=lambda r: r.id):
            parts.append(f"## {rq.id} [OPEN] — {rq.question}")
            if rq.context:
                parts.append(f"  Context: {rq.context}")
            if rq.evidence:
                for idx, ev in enumerate(rq.evidence):
                    prefix = (
                        f"  Evidence {idx + 1}/{len(rq.evidence)} "
                        if len(rq.evidence) > 1
                        else "  Evidence "
                    )
                    parts.append(
                        f"{prefix}({ev.type}): {ev.result[:500] if ev.result else 'pending'}"
                    )
            parts.append("")
        for rq in sorted(resolved_rqs, key=lambda r: r.id):
            status_tag = f"[{rq.status.upper()}]"
            parts.append(f"## {rq.id} {status_tag} — {rq.question}")
            if rq.resolved_to:
                parts.append(f"  Resolved to: {', '.join(rq.resolved_to)}")
            resolution_parts: list[str] = []
            if rq.iteration_resolved is not None:
                resolution_parts.append(f"iteration {rq.iteration_resolved}")
            if rq.resolution_reason:
                resolution_parts.append(rq.resolution_reason)
            if resolution_parts:
                parts.append(f"  Closed: {' — '.join(resolution_parts)}")
            parts.append(
                "  **This RQ is closed. Do not resolve it again or create a WH from it.**"
            )
            parts.append("")

    # Established Results
    parts.append("# Established Results (ER)\n")
    ers = sorted(
        [
            h
            for h in state.hypotheses.values()
            if h.status == HypothesisStatus.ESTABLISHED
        ],
        key=lambda h: h.id,
    )
    for h in ers:
        if h.status == HypothesisStatus.ABANDONED:
            continue  # abandoned go in Dead Ends
        statement_part = f" — {h.statement}" if h.statement else ""
        parts.append(f"## {h.id}{statement_part}\n")
        if h.depends_on:
            parts.append(f"**Depends on:** {', '.join(h.depends_on)}\n")
        if h.derivation:
            parts.append(h.derivation)
            parts.append("")
        if h.evidence:
            for idx, ev in enumerate(h.evidence):
                prefix = (
                    f"**Evidence {idx + 1}/{len(h.evidence)} "
                    if len(h.evidence) > 1
                    else "**Evidence "
                )
                parts.append(f"{prefix}({ev.type}):** {ev.method or 'not specified'}")
                if ev.summary:
                    parts.append(f"  Summary: {ev.summary}")
                if ev.confidence:
                    parts.append(f"  Confidence: {ev.confidence}")
                if ev.result:
                    parts.append(f"  Result: {ev.result[:800]}")
                parts.append("")
        if h.review:
            v = h.review
            parts.append(f"**Review:** {v.verdict}")
            if v.summary:
                parts.append(f"  Summary: {v.summary}")
            if v.details:
                parts.append(f"  Details: {v.details}")
            parts.append("")

    # Working Hypotheses
    parts.append("# Working Hypotheses (WH)\n")
    whs = sorted(
        [h for h in state.hypotheses.values() if h.status == HypothesisStatus.WORKING],
        key=lambda h: h.id,
    )
    for h in whs:
        if h.status == HypothesisStatus.ABANDONED:
            continue  # abandoned go in Dead Ends
        statement_part = f" — {h.statement}" if h.statement else ""
        parts.append(f"## {h.id}{statement_part}\n")
        if h.depends_on:
            parts.append(f"**Depends on:** {', '.join(h.depends_on)}\n")
        if h.derivation:
            parts.append(h.derivation)
            parts.append("")
        # Evidence summary
        if h.evidence:
            for idx, ev in enumerate(h.evidence):
                prefix = (
                    f"**Evidence {idx + 1}/{len(h.evidence)} "
                    if len(h.evidence) > 1
                    else "**Evidence "
                )
                parts.append(f"{prefix}({ev.type}):** {ev.method or 'not specified'}")
                if ev.summary:
                    parts.append(f"  Summary: {ev.summary}")
                if ev.confidence:
                    parts.append(f"  Confidence: {ev.confidence}")
                if ev.result:
                    parts.append(f"  Result: {ev.result[:800]}")
                parts.append("")
        # Review status
        if h.review:
            v = h.review
            parts.append(f"**Review:** {v.verdict}")
            if v.summary:
                parts.append(f"  Summary: {v.summary}")
            if v.details:
                parts.append(f"  Details: {v.details}")
            parts.append("")

    # Dead Ends
    has_dead_ends = bool(state.failed_approaches) or any(
        h.status == HypothesisStatus.ABANDONED for h in state.hypotheses.values()
    )
    parts.append("# Dead Ends\n")
    deduped = _dedup_failed_approaches(state.failed_approaches)
    for fa in deduped:
        parts.append(f"- {fa.description}")
        if fa.reason:
            parts.append(f"  Reason: {fa.reason}")
        if fa.derivation_excerpt:
            parts.append(f"  Derivation: {fa.derivation_excerpt}")
        if fa.related_entities:
            parts.append(f"  Related entities: {', '.join(fa.related_entities)}")
    # Only render abandoned hypotheses not already covered by failed_approaches
    fa_descriptions = {fa.description for fa in deduped}
    for h in sorted(state.hypotheses.values(), key=lambda h: h.id):
        if h.status == HypothesisStatus.ABANDONED:
            desc = f"Abandoned {h.id} — {h.statement}"
            if desc not in fa_descriptions:
                parts.append(f"- {desc}")
    if not has_dead_ends:
        parts.append("(None yet.)")
    parts.append("")

    return "\n".join(parts)


def render_research_state_md(state: ResearchState) -> str:
    """Render RESEARCH_STATE.md from ResearchState."""
    er_ids = sorted(h.id for h in state.hypotheses.values() if h.id.startswith("ER-"))
    meta: dict = {
        "problem_id": "research-session",
        "title": state.title or state.problem_statement[:80],
        "status": state.status,
        "iteration": state.iteration,
    }
    if er_ids:
        meta["verified_results"] = er_ids

    body = _research_state_body(state)
    return render_frontmatter(meta, body)


def _evidence_log_body(state: ResearchState) -> str:
    """Build the body text for an evidence log rendering."""
    parts: list[str] = ["# Evidence Log\n"]

    # Collect all hypotheses with evidence or verification, sorted by iteration
    entries = []
    for h in state.hypotheses.values():
        if h.evidence:
            for ev in h.evidence:
                entries.append(("evidence", ev.iteration or h.iteration_created, h, ev))
        if h.review:
            entries.append(
                ("verification", h.review.iteration or h.iteration_modified, h, None)
            )
    # Also check RQs for evidence — deduplicate when RQ was promoted to a WH
    for rq in state.research_questions.values():
        if rq.evidence:
            promoted = any(
                hid in state.hypotheses and len(state.hypotheses[hid].evidence) > 0
                for hid in rq.resolved_to
            )
            for ev in rq.evidence:
                if promoted:
                    entries.append(
                        ("rq_promoted", ev.iteration or rq.iteration_created, rq, ev)
                    )
                else:
                    entries.append(
                        ("rq_evidence", ev.iteration or rq.iteration_created, rq, ev)
                    )

    entries.sort(key=lambda e: e[1])

    for entry_type, iteration, entity, ev in entries:
        if entry_type == "rq_promoted":
            resolved_ids = ", ".join(entity.resolved_to) if entity.resolved_to else "?"
            ev_label = f" {ev.id}" if ev.id else ""
            parts.append(f"## {entity.id}:{ev_label} Evidence ({ev.type}) → promoted\n")
            parts.append(f"**Question:** {entity.question}")
            parts.append(f"**Result:** {ev.result}")
            parts.append(f"*(Full evidence under {resolved_ids}.)*")
            parts.append(f"**Iteration:** {iteration}\n")
        elif entry_type == "rq_evidence":
            ev_label = f" {ev.id}" if ev.id else ""
            parts.append(f"## {entity.id}:{ev_label} Evidence ({ev.type})\n")
            parts.append(f"**Question:** {entity.question}")
            parts.append(f"**Method:** {ev.method}")
            if ev.summary:
                parts.append(f"**Summary:** {ev.summary}")
            if ev.approach:
                parts.append(f"**Approach:** {ev.approach[:2000]}")
            parts.append(f"**Result:** {ev.result}")
            if ev.confidence:
                parts.append(f"**Confidence:** {ev.confidence}")
            if ev.scripts:
                parts.append(f"**Scripts:** {', '.join(ev.scripts)}")
            parts.append(f"**Iteration:** {iteration}\n")
        elif entry_type == "evidence":
            ev_label = f" {ev.id}" if ev.id else ""
            parts.append(f"## {entity.id}:{ev_label} Evidence ({ev.type})\n")
            parts.append(f"**Statement:** {entity.statement}")
            parts.append(f"**Method:** {ev.method}")
            if ev.summary:
                parts.append(f"**Summary:** {ev.summary}")
            if ev.approach:
                parts.append(f"**Approach:** {ev.approach[:2000]}")
            parts.append(f"**Result:** {ev.result}")
            if ev.confidence:
                parts.append(f"**Confidence:** {ev.confidence}")
            if ev.scripts:
                parts.append(f"**Scripts:** {', '.join(ev.scripts)}")
            if ev.derivation_file:
                parts.append(f"**Derivation file:** `derivations/{ev.derivation_file}`")
            if ev.reasoning:
                parts.append(f"**Reasoning:** {ev.reasoning[:2000]}")
            parts.append(f"**Iteration:** {iteration}\n")
        elif entry_type == "verification":
            v = entity.review
            parts.append(f"## {entity.id}: Review — {v.verdict}\n")
            parts.append(f"**Statement:** {entity.statement}")
            if v.summary:
                parts.append(f"**Summary:** {v.summary}")
            if v.details:
                parts.append(f"**Details:** {v.details}")
            parts.append(f"**Iteration:** {iteration}\n")

    if not entries:
        parts.append("(No evidence or verification recorded yet.)\n")

    return "\n".join(parts)


def render_evidence_log_md(state: ResearchState) -> str:
    """Render EVIDENCE_LOG.md from ResearchState — evidence and verification on hypotheses."""
    body = _evidence_log_body(state)
    # Count entries for frontmatter
    n_entries = (
        sum(len(h.evidence) for h in state.hypotheses.values())
        + sum(1 for h in state.hypotheses.values() if h.review)
        + sum(len(rq.evidence) for rq in state.research_questions.values())
    )
    meta = {"total_entries": n_entries}
    return render_frontmatter(meta, body)


def _critique_log_body(state: ResearchState) -> str:
    """Build the body text for a critique log rendering."""
    active = [c for c in state.critiques.values() if c.status == CritiqueStatus.ACTIVE]
    resolved = [
        c
        for c in state.critiques.values()
        if c.status in (CritiqueStatus.RESOLVED, CritiqueStatus.WITHDRAWN)
    ]

    parts: list[str] = ["# Active Critiques\n"]

    for c in sorted(active, key=lambda c: c.id):
        sev_tag = f"[{c.severity}]"
        parts.append(f"## {c.id} {sev_tag} [UNRESOLVED]\n")
        targets_str = ", ".join(c.targets) if c.targets else "general"
        parts.append(f"**Target:** {targets_str}\n")
        if c.argument:
            parts.append(c.argument)
        parts.append("")

    parts.append("# Resolved Critiques\n")

    for c in sorted(resolved, key=lambda c: c.id):
        sev_tag = f"[{c.severity}]"
        status_tag = f"[{c.status.upper()}]"
        parts.append(f"## {c.id} {sev_tag} {status_tag}\n")
        targets_str = ", ".join(c.targets) if c.targets else "general"
        parts.append(f"**Target:** {targets_str}\n")
        if c.argument:
            parts.append(c.argument)
        if c.resolution:
            parts.append(f"- **Resolution:** {c.resolution}")
        parts.append("")

    if state.critic_clean_reviews:
        parts.append("# Clean Reviews\n")
        for rev in sorted(
            state.critic_clean_reviews, key=lambda r: r.get("iteration", 0)
        ):
            parts.append(
                f"**Iteration {rev.get('iteration', '?')}:** {rev.get('summary', '')}\n"
            )

    return "\n".join(parts)


def render_critique_log_md(state: ResearchState) -> str:
    """Render CRITIQUE_LOG.md from ResearchState."""
    active = [c for c in state.critiques.values() if c.status == CritiqueStatus.ACTIVE]

    meta = {
        "total_critiques": len(state.critiques),
        "unresolved_critiques": len(active),
    }

    body = _critique_log_body(state)
    return render_frontmatter(meta, body)
