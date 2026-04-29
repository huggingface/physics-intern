"""Critique routing, adjudication, and auto-promotion.

Between iterations, the main loop hands newly-filed critiques here:

- ``route_critiques`` splits critiques by target type, invokes the
  adjudicator for ER-targeted ones, and invokes the planner to revise
  strategy when demotions or strategy critiques occurred.
- ``adjudicate_er_critique`` runs the adjudicator on a single ER critique
  and demotes the ER (with cascade to dependents) when the critique is valid.
- ``invoke_planner_revision`` runs the planner in revise mode and applies
  the returned strategy, sanity-check, and entity-action updates.
- ``auto_promote`` upgrades a VERIFIED WH to ER once its dependencies are
  established, cascading to other eligible WHs.

All functions take explicit ``research_state`` / ``loop_state`` /
``workspace`` / agent references rather than reading from an engine instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .console import console
from .state_transitions import demote_hypothesis, promote_hypothesis
from .task import Task, TaskType
from .utils.categories import CompensationCategory as CC
from .workspace import log_scaffold_event

if TYPE_CHECKING:
    from .agents.adjudicator import AdjudicatorAgent
    from .agents.planner import PlannerAgent
    from .loop_state import LoopState
    from .research_state import Critique, ResearchState
    from .workspace import WorkspaceManager


def route_critiques(
    research_state: ResearchState,
    loop_state: LoopState,
    iteration: int,
    workspace: WorkspaceManager,
    planner: PlannerAgent,
    adjudicator: AdjudicatorAgent,
    on_round,
) -> None:
    """Route critic findings to specialist agents (synchronous).

    Phase 1: Adjudicate ER-targeted critiques via the adjudicator.
    Phase 2: Route strategy/coordination critiques (and any ER demotions
             from phase 1) to the planner for strategy revision.
    """
    from .research_state import CritiqueStatus

    new_critiques = [
        c
        for c in research_state.critiques.values()
        if c.iteration_filed == iteration and c.status == CritiqueStatus.ACTIVE
    ]
    if not new_critiques:
        return

    # Separate by target_type
    er_critiques = [c for c in new_critiques if c.target_type == "er"]
    strategy_critiques = [
        c
        for c in new_critiques
        if c.target_type in ("strategy", "coordination", "sanity_check")
    ]
    untyped = [
        c
        for c in new_critiques
        if c.target_type not in ("er", "strategy", "coordination", "sanity_check")
    ]

    # Warn and auto-resolve untyped critiques
    for c in untyped:
        console.print(f"  [yellow]{c.id} has no target_type — auto-resolving[/yellow]")
        c.status = CritiqueStatus.RESOLVED
        c.resolution = "Auto-resolved: missing target_type in critic output"
        c.resolution_type = "dismissed"
        c.iteration_resolved = iteration

    # Phase 1: Adjudicate ER-targeted critiques
    er_demotions: list[dict] = []
    for crit in er_critiques:
        try:
            result = adjudicate_er_critique(
                crit,
                research_state,
                loop_state,
                iteration,
                workspace,
                adjudicator,
                on_round,
            )
            if result and result.get("demoted"):
                er_demotions.append(result)
        except Exception as exc:
            console.print(f"  [red]Adjudication failed for {crit.id}: {exc}[/red]")
            log_scaffold_event(
                workspace.root,
                iteration,
                CC.STATE_INVARIANTS,
                "adjudication_error",
                f"{crit.id}: {exc}",
            )

    # Phase 2: Strategy assessment (if demotions or strategy critiques)
    if er_demotions or strategy_critiques:
        try:
            invoke_planner_revision(
                strategy_critiques,
                er_demotions,
                research_state,
                loop_state,
                iteration,
                workspace,
                planner,
                on_round,
            )
        except Exception as exc:
            console.print(f"  [red]Planner revision failed: {exc}[/red]")
            log_scaffold_event(
                workspace.root,
                iteration,
                CC.STATE_INVARIANTS,
                "planner_revision_error",
                str(exc),
            )


def adjudicate_er_critique(
    crit: Critique,
    research_state: ResearchState,
    loop_state: LoopState,
    iteration: int,
    workspace: WorkspaceManager,
    adjudicator: AdjudicatorAgent,
    on_round,
) -> dict | None:
    """Invoke the adjudicator to evaluate an ER-targeted critique.

    Returns dict with demotion info if ER was overturned, else None.
    """
    from .research_state import CritiqueStatus, HypothesisStatus, ResearchQuestion

    target_id = crit.targets[0] if crit.targets else None
    if not target_id or target_id not in research_state.hypotheses:
        console.print(
            f"  [dim]{crit.id} targets unknown entity {target_id} — dismissing[/dim]"
        )
        crit.status = CritiqueStatus.RESOLVED
        crit.resolution = f"Target {target_id} not found"
        crit.resolution_type = "dismissed"
        crit.iteration_resolved = iteration
        return None

    console.print(
        f"  [cyan]Adjudicator[/cyan] evaluating {crit.id} against {target_id}..."
    )
    adjud_task = Task(
        task_id=f"ADJUD-{iteration:03d}-{crit.id}",
        task_type=TaskType.ADJUDICATE,
        assigned_to="adjudicator",
        iteration=iteration,
        target_claim=target_id,
        critique_argument=crit.argument,
    )
    adjudicator.research_state = research_state
    adjudicator.run(adjud_task, iteration, on_round=on_round)
    result = adjudicator.adjudication_result

    if not result:
        console.print(f"  [yellow]{crit.id}: adjudicator returned no result[/yellow]")
        return None

    adjudication = result.get("adjudication", "needs_evidence")
    reasoning = result.get("reasoning", "")[:200]

    if adjudication == "valid":
        from .research_state import FailedApproach

        # Collect dependents before first demotion (normalize_references
        # rewrites depends_on from ER-NNN to WH-NNN after demotion)
        dependent_ids = [
            hid
            for hid, h in research_state.hypotheses.items()
            if h.status == HypothesisStatus.ESTABLISHED and target_id in h.depends_on
        ]
        # Demote ER → WH and auto-abandon
        console.print(f"  [red]{crit.id} VALID — demoting {target_id}[/red]")
        new_id = demote_hypothesis(research_state, target_id)
        if new_id:
            h = research_state.hypotheses[new_id]
            h.status = HypothesisStatus.ABANDONED
            h.review = None  # stale VERIFIED review must not trigger re-promotion
            h.iteration_modified = iteration
            research_state.failed_approaches.append(
                FailedApproach(
                    description=f"Overturned {target_id} — {h.statement}",
                    reason=f"Adjudicator ruled critique {crit.id} valid: {reasoning}",
                    related_entities=[new_id],
                    derivation_excerpt=(h.derivation[:300] if h.derivation else ""),
                    iteration=iteration,
                )
            )
        # Cascade: demote and auto-abandon dependents
        for dep_id in dependent_ids:
            console.print(
                f"  [red]Cascade: demoting {dep_id} (depends on {target_id})[/red]"
            )
            dep_new_id = demote_hypothesis(research_state, dep_id)
            if dep_new_id:
                dep_h = research_state.hypotheses[dep_new_id]
                dep_h.status = HypothesisStatus.ABANDONED
                dep_h.review = None  # prevent stale re-promotion
                dep_h.iteration_modified = iteration
                research_state.failed_approaches.append(
                    FailedApproach(
                        description=f"Cascade from overturned {target_id} — {dep_h.statement}",
                        reason=f"Depends on {target_id} which was overturned",
                        related_entities=[dep_new_id],
                        iteration=iteration,
                    )
                )
            loop_state.pending_system_events.append(
                f"{dep_id} DEMOTED and ABANDONED (depends on overturned {target_id})"
            )
        crit.status = CritiqueStatus.RESOLVED
        crit.resolution = f"Adjudicator ruled valid: {reasoning}"
        crit.resolution_type = "accepted"
        crit.iteration_resolved = iteration
        loop_state.pending_system_events.append(
            f"{target_id} OVERTURNED and ABANDONED: {crit.id} ruled valid by adjudicator."
        )
        log_scaffold_event(
            workspace.root,
            iteration,
            CC.STATE_INVARIANTS,
            "er_demotion",
            f"{target_id} overturned by {crit.id}",
        )
        return {"demoted": target_id, "critique": crit.id, "reasoning": reasoning}

    elif adjudication == "invalid":
        console.print(f"  [green]{crit.id} INVALID — {target_id} stands[/green]")
        counter = result.get("counter_argument", "")[:200]
        crit.status = CritiqueStatus.RESOLVED
        crit.resolution = f"Adjudicator ruled invalid: {counter}"
        crit.resolution_type = "dismissed"
        crit.iteration_resolved = iteration
        loop_state.pending_system_events.append(
            f"{crit.id} against {target_id} DISMISSED by adjudicator."
        )
        return None

    else:  # needs_evidence
        console.print(f"  [yellow]{crit.id} NEEDS EVIDENCE — creating RQ[/yellow]")
        scope = result.get("investigation_scope", "Investigate the disputed claim.")
        num = research_state.next_entity_num()
        rq_id = f"RQ-{num:03d}"
        research_state.research_questions[rq_id] = ResearchQuestion(
            id=rq_id,
            question=scope,
            context=f"Created by adjudicator for {crit.id} targeting {target_id}.",
            iteration_created=iteration,
        )
        loop_state.pending_system_events.append(
            f"{crit.id}: adjudicator needs evidence — created {rq_id}."
        )
        return None


def invoke_planner_revision(
    strategy_critiques: list[Critique],
    er_demotions: list[dict],
    research_state: ResearchState,
    loop_state: LoopState,
    iteration: int,
    workspace: WorkspaceManager,
    planner: PlannerAgent,
    on_round,
) -> None:
    """Invoke the planner in revise mode to assess strategy after critiques/demotions."""
    from .research_state import CritiqueStatus, HypothesisStatus

    # Build trigger text
    trigger_parts: list[str] = []
    for d in er_demotions:
        trigger_parts.append(
            f"ER {d['demoted']} was overturned by critique {d['critique']}. "
            f"Adjudicator reasoning: {d['reasoning']}"
        )
    for c in strategy_critiques:
        trigger_parts.append(
            f"Critique {c.id} [{c.severity}] targeting {c.target_type}: {c.argument}"
        )
    trigger_text = "\n\n".join(trigger_parts)

    console.print("  [cyan]Planner[/cyan] revising strategy...")
    revise_task = Task(
        task_id=f"PLAN-REVISE-{iteration:03d}",
        task_type=TaskType.PLAN_REVISE,
        assigned_to="planner",
        iteration=iteration,
        body=trigger_text,
    )
    planner.research_state = research_state
    planner.run(revise_task, iteration, on_round=on_round)

    # Apply results
    if planner.parsed_strategy:
        research_state.strategy = planner.parsed_strategy
        console.print("  [green]Strategy updated[/green]")

    if planner.parsed_sanity_checks is not None:
        from .research_state import SanityCheck

        new_checks: list[SanityCheck] = []
        for item in planner.parsed_sanity_checks:
            if isinstance(item, dict):
                existing_id = item.get("id", "")
                predicate = item.get("predicate", str(item))
                rationale = item.get("rationale", "")
                if existing_id and existing_id.startswith("SC-"):
                    new_checks.append(
                        SanityCheck(
                            id=existing_id, predicate=predicate, rationale=rationale
                        )
                    )
                else:
                    sc_num = research_state.next_sc_num()
                    # Account for checks already added in this batch
                    while any(c.id == f"SC-{sc_num:03d}" for c in new_checks):
                        sc_num += 1
                    new_checks.append(
                        SanityCheck(
                            id=f"SC-{sc_num:03d}",
                            predicate=predicate,
                            rationale=rationale,
                        )
                    )
            elif isinstance(item, str) and item.strip():
                sc_num = research_state.next_sc_num()
                while any(c.id == f"SC-{sc_num:03d}" for c in new_checks):
                    sc_num += 1
                new_checks.append(
                    SanityCheck(id=f"SC-{sc_num:03d}", predicate=item.strip())
                )
        research_state.sanity_checks = new_checks
        console.print(f"  [dim]Sanity checks updated ({len(new_checks)} checks)[/dim]")

    if planner.parsed_entity_actions:
        for action in planner.parsed_entity_actions:
            eid = action.get("id", "")
            act = action.get("action", "keep")
            reason = action.get("reason", "")
            if act == "keep":
                concern = action.get("concern", "")
                if concern and eid:
                    loop_state.pending_system_events.append(
                        f"PLANNER CONCERN on {eid}: {concern}"
                    )
            elif act == "abandon" and eid in research_state.hypotheses:
                from .research_state import FailedApproach

                h = research_state.hypotheses[eid]
                if h.status == HypothesisStatus.ABANDONED:
                    console.print(f"  [dim]{eid} already abandoned, skipping[/dim]")
                    continue
                h.status = HypothesisStatus.ABANDONED
                h.iteration_modified = iteration
                research_state.failed_approaches.append(
                    FailedApproach(
                        description=f"Abandoned {eid} — {h.statement}",
                        reason=f"Planner revision: {reason}",
                        related_entities=[eid],
                        derivation_excerpt=(h.derivation[:300] if h.derivation else ""),
                        iteration=iteration,
                    )
                )
                loop_state.pending_system_events.append(
                    f"{eid} ABANDONED by planner revision: {reason}"
                )
                console.print(f"  [red]{eid} abandoned: {reason[:60]}[/red]")

    rationale = planner.parsed_revision_rationale or "No rationale provided."

    # Resolve strategy/coordination critiques using planner's assessments
    assessments_by_id: dict[str, dict] = {}
    if planner.parsed_critique_assessments:
        for a in planner.parsed_critique_assessments:
            assessments_by_id[a.get("id", "")] = a

    for c in strategy_critiques:
        assessment = assessments_by_id.get(c.id)
        c.status = CritiqueStatus.RESOLVED
        c.iteration_resolved = iteration

        if assessment and assessment.get("verdict") == "dismiss":
            dismiss_reason = assessment.get("reason", "Dismissed by planner")[:200]
            c.resolution = f"Dismissed by planner: {dismiss_reason}"
            c.resolution_type = "dismissed"
            console.print(
                f"  [yellow]{c.id} dismissed by planner: {dismiss_reason[:60]}[/yellow]"
            )
        else:
            c.resolution = f"Addressed in strategy revision: {rationale[:120]}"
            c.resolution_type = "accepted"
            console.print(f"  [green]{c.id} accepted by planner[/green]")

    accepted_ids = [c.id for c in strategy_critiques if c.resolution_type == "accepted"]
    dismissed_ids = [
        c.id for c in strategy_critiques if c.resolution_type == "dismissed"
    ]
    label_parts: list[str] = []
    if accepted_ids:
        label_parts.append(f"accepted: {', '.join(accepted_ids)}")
    if dismissed_ids:
        label_parts.append(f"dismissed: {', '.join(dismissed_ids)}")
    event_label = (
        f"STRATEGY REVISED ({'; '.join(label_parts)})"
        if label_parts
        else "STRATEGY REVISED"
    )
    loop_state.pending_system_events.append(f"{event_label}: {rationale}")

    log_scaffold_event(
        workspace.root,
        iteration,
        CC.STATE_INVARIANTS,
        "strategy_revision",
        rationale[:200],
    )


def auto_promote(
    research_state: ResearchState,
    wh_id: str,
    iteration: int,
    workspace: WorkspaceManager,
) -> None:
    """Auto-promote a VERIFIED WH to ER if dependencies are satisfied.

    After promotion, cascades: scans remaining WHs for VERIFIED ones
    whose dependencies are now all established, and promotes those too.
    """
    from .research_state import HypothesisStatus, Verdict

    # Seed the cascade with the initial candidate
    candidates = [wh_id]
    while candidates:
        current_id = candidates.pop(0)
        if current_id not in research_state.hypotheses:
            continue
        h = research_state.hypotheses[current_id]
        # Must be WORKING and VERIFIED to promote
        if h.status != HypothesisStatus.WORKING:
            continue
        if not h.review or h.review.verdict != Verdict.VERIFIED:
            continue
        unestablished = research_state.unestablished_dependencies(current_id)
        if unestablished:
            console.print(
                f"  [dim]Auto-promote skipped for {current_id} "
                f"(unestablished deps: {', '.join(unestablished)})[/dim]"
            )
            continue
        er_id = promote_hypothesis(research_state, current_id, iteration)
        log_scaffold_event(
            workspace.root,
            iteration,
            CC.STATE_INVARIANTS,
            "auto_promote",
            f"{current_id} → {er_id}",
        )
        console.print(f"  [bold green]{current_id} → {er_id}[/] auto-promoted")
        # Cascade: find VERIFIED WHs that might now have all deps met
        for hid, hyp in research_state.hypotheses.items():
            if (
                hid.startswith("WH-")
                and hyp.review
                and hyp.review.verdict == Verdict.VERIFIED
                and hid not in candidates
            ):
                candidates.append(hid)
