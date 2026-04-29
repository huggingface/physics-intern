"""Inter-iteration loop state and orchestrator context banners.

``LoopState`` is the mutable scratch space the main loop carries across
iterations: dispatch history, consumed-once feedback banners, failure
counters. The render functions in this module build the orchestrator's
context suffix from that state.

All functions here are pure — they read state and return strings (or mutate
the fields explicitly documented). No agents or workspace side-effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .research_state import Verdict

if TYPE_CHECKING:
    from .config import Config
    from .research_state import ResearchState
    from .task import Task


@dataclass
class DispatchRecord:
    """One-line record of what was dispatched in a given iteration."""

    iteration: int
    task_type: str  # "compute", "review", "critique", etc.
    target: str | None  # "WH-001", "RQ-003", or None
    outcome: str  # "evidence (exact)", "REFUTED", "3 critique(s)", etc.


@dataclass
class LoopState:
    """Inter-iteration state for the main research loop."""

    claim_failure_count: dict[str, int] = field(default_factory=dict)
    last_content_iteration: int = 0
    consecutive_termination_blocks: int = 0
    # Consumed-once feedback accumulators (cleared after orchestrator reads them)
    pending_violations: list = field(default_factory=list)
    pending_termination_blockers: list[str] = field(default_factory=list)
    pending_compute_verdicts: list[dict] = field(default_factory=list)
    pending_verified_results: list[dict] = field(default_factory=list)
    pending_explore_results: list[dict] = field(default_factory=list)
    agent_failures: list[dict] = field(default_factory=list)
    last_verified_review_iteration: int = 0
    pending_system_events: list[str] = field(default_factory=list)
    # Persistent dispatch history (never cleared)
    dispatch_history: list[DispatchRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dispatch record building
# ---------------------------------------------------------------------------


def append_dispatch_record(
    loop_state: LoopState,
    research_state: ResearchState,
    task: Task,
    iteration: int,
) -> None:
    """Derive an outcome from authoritative state and append a DispatchRecord."""
    from .task import TaskType

    tt = task.task_type
    target = task.target_claim or None

    if tt in (TaskType.RESEARCH, TaskType.COMPUTE):
        ev = None
        if target and target in research_state.research_questions:
            evs = research_state.research_questions[target].evidence
            ev = evs[-1] if evs else None
        elif target and target in research_state.hypotheses:
            evs = research_state.hypotheses[target].evidence
            ev = evs[-1] if evs else None
        elif target and target in research_state.critiques:
            evs = research_state.critiques[target].evidence
            ev = evs[-1] if evs else None
        if ev and ev.result:
            outcome = f"evidence ({ev.confidence})" if ev.confidence else "evidence"
        else:
            outcome = "no evidence"

    elif tt == TaskType.REVIEW:
        h = None
        if target and target in research_state.hypotheses:
            h = research_state.hypotheses[target]
        elif target and target.startswith("WH-"):
            er_id = f"ER-{target.split('-')[1]}"
            if er_id in research_state.hypotheses:
                h = research_state.hypotheses[er_id]
        if h and h.review:
            outcome = f"{h.review.verdict} → {h.id}"
        else:
            outcome = "no review produced"

    elif tt == TaskType.CRITIQUE:
        recent = [
            c
            for c in research_state.critiques.values()
            if c.iteration_filed == iteration
        ]
        if recent:
            outcome = f"{len(recent)} critique(s)"
        else:
            outcome = "no critiques"

    elif tt == TaskType.TERMINATE:
        outcome = "blocked"

    else:
        outcome = "completed"

    loop_state.dispatch_history.append(
        DispatchRecord(
            iteration=iteration,
            task_type=tt.value,
            target=target,
            outcome=outcome,
        )
    )


# ---------------------------------------------------------------------------
# Pending work summary
# ---------------------------------------------------------------------------


def render_pending_work(research_state: ResearchState) -> str:
    """Render a summary of open RQs, working WHs, and dangling WHs."""
    if not research_state:
        return ""

    lines: list[str] = []
    whs = research_state.working_hypotheses()
    if whs:
        wh_items = []
        for h in whs:
            if h.evidence:
                status = (
                    h.review.verdict.upper()
                    if h.review
                    else f"has {len(h.evidence)} evidence, PENDING REVIEW"
                )
            else:
                status = "no evidence"
            wh_items.append(f"{h.id} ({status})")
        lines.append(f"  WH: {', '.join(wh_items)}")

        # Dangling WHs: REFUTED or INCONCLUSIVE, still WORKING
        dangling = [
            h
            for h in whs
            if h.review and h.review.verdict in (Verdict.REFUTED, Verdict.INCONCLUSIVE)
        ]
        if dangling:
            lines.append(
                "  >>> ATTENTION: resolve these WHs before dispatching to RQs <<<"
            )
            for h in dangling:
                rc_note = f", refuted {h.refuted_count}x" if h.refuted_count else ""
                lines.append(
                    f"    {h.id}: {h.review.verdict}{rc_note}"
                    " — gather new evidence or abandon"
                )

    open_rqs = research_state.open_research_questions()
    if open_rqs:
        rq_items = []
        for rq in open_rqs:
            ev = f"{len(rq.evidence)} evidence" if rq.evidence else "no evidence"
            rq_items.append(f"{rq.id} ({ev})")
        lines.append(f"  Open RQs: {', '.join(rq_items)}")

    if not lines:
        return ""
    return ">>> PENDING WORK <<<\n" + "\n".join(lines) + "\n>>> END PENDING WORK <<<\n"


# ---------------------------------------------------------------------------
# Context suffix (orchestrator banners)
# ---------------------------------------------------------------------------


def build_context_suffix(
    loop_state: LoopState,
    research_state: ResearchState,
    config: Config,
    iteration: int,
) -> tuple[str, str]:
    """Build the orchestrator's inter-iteration banners.

    Returns ``(suffix, dispatch_history_text)``. The caller assigns
    ``dispatch_history_text`` to the orchestrator separately (it feeds into
    the research-state section, not the trailing banners).

    Consumed-once banners are cleared from ``loop_state`` after rendering.
    """
    dispatch_history_text = ""
    if loop_state.dispatch_history:
        cutoff = max(iteration - 4, 0)
        recent = [r for r in loop_state.dispatch_history if r.iteration >= cutoff]
        omitted = len(loop_state.dispatch_history) - len(recent)
        dh_lines = ["<tasks_dispatch_history>"]
        if omitted > 0:
            dh_lines.append(f"(...{omitted} earlier dispatch(es) omitted)")
        for rec in recent:
            target_str = f" → {rec.target}" if rec.target else ""
            dh_lines.append(
                f"Iter {rec.iteration}: {rec.task_type}{target_str} | {rec.outcome}"
            )
        dh_lines.append("</tasks_dispatch_history>")
        dispatch_history_text = "\n".join(dh_lines)

    lines: list[str] = []
    if loop_state.pending_violations:
        lines.append(">>> POST-INTEGRATION VIOLATIONS <<<")
        for v in loop_state.pending_violations:
            lines.append(f"  [{v.severity}] {v.check}: {v.message}")
        lines.append(">>> END VIOLATIONS <<<\n")
        loop_state.pending_violations.clear()
    if loop_state.pending_termination_blockers:
        lines.append(">>> TERMINATION BLOCKED — YOU CANNOT TERMINATE YET <<<")
        lines.append("Your previous terminate request was REJECTED for these reasons:")
        for b in loop_state.pending_termination_blockers:
            lines.append(f"  - {b}")
        lines.append(
            "Do request termination again until you have addressed ALL blockers above."
        )
        lines.append("")
        lines.append("Pre-dispatch checklist (verify before retrying termination):")
        lines.append(
            "1. Every FILL IN placeholder in the answer template has a concrete ER."
        )
        lines.append(
            "2. ER expressions are explicit closed-form SymPy (no abstract operators or opaque functions)."
        )
        lines.append(
            "3. MCQ answers are a concrete letter from the given set, not prose."
        )
        lines.append("4. Return types match the template (tuple elements, etc.).")
        lines.append(">>> END TERMINATION BLOCKERS <<<\n")
        loop_state.pending_termination_blockers.clear()
    if loop_state.pending_explore_results:
        lines.append(">>> EVIDENCE RESULTS (previous iteration) <<<")
        for r in loop_state.pending_explore_results:
            ev_label = f" [{r['evidence_id']}]" if r.get("evidence_id") else ""
            provenance = (
                f"  [from {r['task_id']}: {r['task_type']} on {r['target_id']}]"
            )
            lines.append(
                f"-{ev_label} {r['target_id']}: {r['description']}  [{r['confidence']}]{provenance}"
            )
            if r.get("result"):
                lines.append(f"  Result: {r['result']}")
            _is_failure = r.get("result", "").startswith(
                ("Agent produced no exit tool call", "Failed to parse structured")
            )
            if _is_failure:
                lines.append(
                    "  NOTE: This evidence is from a failed agent run — do NOT treat it as usable evidence."
                )
            # --- Evidence accumulation nudges ---
            tid = r["target_id"]
            count = r.get("evidence_count", 0)
            types = r.get("evidence_types", {})
            is_rq = r.get("target_is_rq", False)
            if is_rq and not _is_failure:
                lines.append(
                    f"  -> ACTION NEEDED: {tid} now has {count} evidence item(s) on a Research Question."
                    " Consider promoting to a Working Hypothesis (add_hypothesis) so it undergoes adversarial review."
                )
            rq_cap = config.rq_evidence_cap
            if count >= rq_cap and is_rq:
                lines.append(
                    f"  >> BLOCKED: {tid} has {count} evidence items (cap={rq_cap}) WITHOUT a Working Hypothesis."
                    " dispatch_researcher / dispatch_computer WILL BE REJECTED until you"
                    " promote this RQ to a WH (add_hypothesis) or resolve/abandon it."
                )
            if count >= 3 and len(types) == 1:
                only_type = next(iter(types))
                alt = "researcher" if only_type == "compute" else "computer"
                lines.append(
                    f"  >> NOTE: All {count} evidence items on {tid} are type '{only_type}'."
                    f" Consider dispatching a {alt} for a different analytical perspective."
                )
        lines.append(">>> END EVIDENCE RESULTS <<<\n")
        loop_state.pending_explore_results.clear()
    if loop_state.pending_verified_results:
        lines.append(">>> VERIFIED HYPOTHESES (previous iteration) <<<")
        for v in loop_state.pending_verified_results:
            provenance = f"  [from {v['task_id']}]" if v.get("task_id") else ""
            lines.append(f"- {v['claim']} VERIFIED by reviewer{provenance}")
            if v.get("reasoning"):
                lines.append(f"  Reasoning: {v['reasoning']}")
        lines.append(">>> END VERIFIED HYPOTHESES <<<\n")
        loop_state.pending_verified_results.clear()
    if loop_state.pending_compute_verdicts:
        lines.append(">>> VERIFICATION RESULTS (previous iteration) <<<")
        for v in loop_state.pending_compute_verdicts:
            provenance = f"  [from {v['task_id']}]" if v.get("task_id") else ""
            lines.append(f"- {v['verdict']}: {v['claim'][:120]}{provenance}")
            lines.append(f"  Attempt {v['attempt']}/{config.stall_recompute_limit}")
            if v.get("notes"):
                lines.append(f"  Notes: {v['notes']}")
            if v.get("details"):
                lines.append(f"  Details: {v['details']}")
            if v["attempt"] >= config.stall_recompute_limit:
                lines.append(
                    "  STALLED — do NOT schedule another review. Try alternative evidence."
                )
        lines.append(">>> END VERIFICATION RESULTS <<<\n")
        loop_state.pending_compute_verdicts.clear()
    if loop_state.agent_failures:
        lines.append(">>> AGENT FAILURES (previous iteration) <<<")
        for f in loop_state.agent_failures:
            lines.append(
                f"  - [{f['task_id']}] {f['agent']}: {f['event']}. {f['detail']}"
            )
        lines.append(">>> END AGENT FAILURES <<<\n")
        loop_state.agent_failures.clear()
    # System events from critique routing (ER demotions, strategy revisions, etc.)
    if loop_state.pending_system_events:
        lines.append(">>> SYSTEM EVENTS (between iterations) <<<")
        for event in loop_state.pending_system_events:
            lines.append(f"- {event}")
        lines.append(">>> END SYSTEM EVENTS <<<\n")
        loop_state.pending_system_events.clear()
    # Pending work summary — always present so the orchestrator sees current state
    pending = render_pending_work(research_state)
    if pending:
        lines.append(pending)
    return ("\n".join(lines), dispatch_history_text)
