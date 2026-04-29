"""Post-integration validation and termination gates for OpenDirac."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from .research_state import CritiqueStatus, HypothesisStatus, Verdict
from .state_transitions import demote_hypothesis
from .utils.categories import CompensationCategory as CC
from .workspace import log_scaffold_event

if TYPE_CHECKING:
    from .config import Config
    from .metrics import MetricsTracker
    from .research_state import ResearchState
    from .workspace import WorkspaceManager


class ViolationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Violation:
    check: str
    severity: ViolationSeverity
    message: str
    detail: str = ""


# ---------------------------------------------------------------------------
# ER/WH ID pattern (used by multiple checks)
# ---------------------------------------------------------------------------
_ER_WH_ID_RE = re.compile(r"\b(?:ER|WH)-\d+\b")


# ---------------------------------------------------------------------------
# Individual check functions — all operate on ResearchState
# ---------------------------------------------------------------------------


def check_er_demotion_safety(research_state: ResearchState) -> list[Violation]:
    """Demote ER-NNN hypotheses when verification verdict is REFUTED."""
    violations: list[Violation] = []

    for hid, h in list(research_state.hypotheses.items()):
        if not hid.startswith("ER-"):
            continue
        # Check for REFUTED verification without a superseding VERIFIED
        if h.review and h.review.verdict == Verdict.REFUTED:
            new_id = demote_hypothesis(research_state, hid)
            if new_id:
                violations.append(
                    Violation(
                        check="er_demotion_safety",
                        severity=ViolationSeverity.WARNING,
                        message=f"{hid} has REFUTED review — demoted to {new_id}",
                        detail=hid,
                    )
                )

    return violations


def check_phantom_labels(research_state: ResearchState) -> list[Violation]:
    """Strip unsubstantiated VERIFIED labels from hypothesis derivations."""
    violations: list[Violation] = []

    # Build set of hypothesis IDs that have VERIFIED review
    verified_ids: set[str] = set()
    for hid, h in research_state.hypotheses.items():
        if h.review and h.review.verdict == Verdict.VERIFIED:
            verified_ids.add(hid)

    for hid, h in research_state.hypotheses.items():
        if not h.derivation or "VERIFIED" not in h.derivation:
            continue
        new_lines = []
        changed = False
        for line in h.derivation.splitlines():
            if "VERIFIED" in line and not line.strip().startswith("##"):
                ids_in_line = _ER_WH_ID_RE.findall(line)
                if ids_in_line:
                    backed = any(id_ in verified_ids for id_ in ids_in_line)
                    if not backed:
                        line = line.replace("VERIFIED", "[unverified]")
                        changed = True
                        violations.append(
                            Violation(
                                check="phantom_labels",
                                severity=ViolationSeverity.ERROR,
                                message="Unsubstantiated VERIFIED label stripped",
                                detail=f"{hid}: {', '.join(ids_in_line)}",
                            )
                        )
            new_lines.append(line)
        if changed:
            h.derivation = "\n".join(new_lines)

    return violations


def check_stale_unverified_labels(research_state: ResearchState) -> list[Violation]:
    """Promote [unverified] labels to VERIFIED when backed by verification."""
    violations: list[Violation] = []

    # Build set of hypothesis IDs that have VERIFIED review
    verified_ids: set[str] = set()
    for hid, h in research_state.hypotheses.items():
        if h.review and h.review.verdict == Verdict.VERIFIED:
            verified_ids.add(hid)
            # Also add the ER form if WH was verified and has been promoted
            if hid.startswith("WH-"):
                num = hid.split("-")[1]
                er_form = f"ER-{num}"
                if er_form in research_state.hypotheses:
                    verified_ids.add(er_form)
            # Also add the WH form so derivations referencing old ID are matched
            if hid.startswith("ER-"):
                num = hid.split("-")[1]
                verified_ids.add(f"WH-{num}")

    if not verified_ids:
        return []

    for hid, h in research_state.hypotheses.items():
        if not h.derivation or "[unverified]" not in h.derivation.lower():
            continue
        new_lines = []
        changed = False
        for line in h.derivation.splitlines():
            if "[unverified]" in line.lower():
                ids_in_line = _ER_WH_ID_RE.findall(line)
                if ids_in_line and any(id_ in verified_ids for id_ in ids_in_line):
                    line = line.replace("[unverified]", "VERIFIED")
                    line = line.replace("[Unverified]", "VERIFIED")
                    for wh in [i for i in ids_in_line if i.startswith("WH-")]:
                        num = wh.split("-")[1]
                        er_form = f"ER-{num}"
                        if er_form in research_state.hypotheses:
                            line = line.replace(wh, er_form)
                    changed = True
                    violations.append(
                        Violation(
                            check="stale_unverified_labels",
                            severity=ViolationSeverity.WARNING,
                            message=f"Promoted [unverified] → VERIFIED for {', '.join(ids_in_line)}",
                            detail=f"{hid}: {', '.join(ids_in_line)}",
                        )
                    )
            new_lines.append(line)
        if changed:
            h.derivation = "\n".join(new_lines)

    return []


def check_critique_resolution_consistency(
    research_state: ResearchState,
) -> list[Violation]:
    """Check that resolved critiques are consistent with current state."""
    violations: list[Violation] = []

    _LABEL_KEYWORDS = re.compile(
        r"label|inconsisten|rename|mislabel|header", re.IGNORECASE
    )

    for crit_id, crit in research_state.critiques.items():
        if crit.status != CritiqueStatus.RESOLVED:
            continue
        if not crit.targets:
            continue

        body_text = crit.argument or ""
        is_label_critique = bool(_LABEL_KEYWORDS.search(body_text))

        for tid in crit.targets:
            if "-" not in tid:
                continue
            num = tid.split("-")[1]
            wh_form = f"WH-{num}"
            er_form = f"ER-{num}"

            if (
                tid not in research_state.hypotheses
                and wh_form not in research_state.hypotheses
                and er_form not in research_state.hypotheses
            ):
                violations.append(
                    Violation(
                        check="critique_resolution_consistency",
                        severity=ViolationSeverity.WARNING,
                        message=(
                            f"Resolved {crit_id} targets {tid} which no longer "
                            f"exists in research state"
                        ),
                        detail=f"{crit_id}:{tid}",
                    )
                )

            if (
                is_label_critique
                and wh_form in research_state.hypotheses
                and er_form in research_state.hypotheses
            ):
                violations.append(
                    Violation(
                        check="critique_resolution_consistency",
                        severity=ViolationSeverity.WARNING,
                        message=(
                            f"Resolved {crit_id} (label critique) but {wh_form} and "
                            f"{er_form} still co-exist in research state"
                        ),
                        detail=f"{crit_id}:{wh_form}+{er_form}",
                    )
                )

    return violations


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_DEFAULT_CHECKS = [
    check_er_demotion_safety,
    check_phantom_labels,
    check_stale_unverified_labels,
    check_critique_resolution_consistency,
]


def validate_post_integration(
    research_state: ResearchState,
    *,
    iteration: int = 0,
    workspace: WorkspaceManager | None = None,
) -> list[Violation]:
    """Run all post-integration invariant checks."""
    violations: list[Violation] = []
    for check in _DEFAULT_CHECKS:
        check_violations = check(research_state)
        if (
            check_violations
            and workspace
            and hasattr(workspace, "root")
            and workspace.root
        ):
            for v in check_violations:
                log_scaffold_event(
                    workspace.root, iteration, CC.STATE_INVARIANTS, v.check, v.message
                )
        violations.extend(check_violations)
    return violations


# ---------------------------------------------------------------------------
# Termination gate
# ---------------------------------------------------------------------------


def can_terminate(
    workspace: WorkspaceManager,
    config: Config,
    metrics: MetricsTracker,
    problem_meta: dict | None = None,
    *,
    research_state: ResearchState,
) -> tuple[bool, list[str]]:
    """Check preconditions before allowing the research loop to exit."""
    from .research_state import RQStatus

    blockers: list[str] = []

    # Gate 1: At least one critic pass if verified results exist
    has_verified = any(
        h.review and h.review.verdict == Verdict.VERIFIED
        for h in research_state.hypotheses.values()
    )
    if has_verified and metrics.last_critic_iteration == 0:
        blockers.append(
            "No critic pass has occurred yet. "
            "A critic pass is triggered automatically after a VERIFIED review."
        )

    # Gate 2: All RQs and WHs must be resolved before termination
    for rq in research_state.research_questions.values():
        if rq.status == RQStatus.OPEN:
            blockers.append(
                f"{rq.id} is still OPEN. "
                "Call abandon_research_question to close it before terminating."
            )

    for h in research_state.hypotheses.values():
        if h.status == HypothesisStatus.WORKING:
            if h.review and h.review.verdict == Verdict.VERIFIED:
                unest = research_state.unestablished_dependencies(h.id)
                if unest:
                    blockers.append(
                        f"{h.id} has VERIFIED review but unestablished dependencies "
                        f"({', '.join(unest)}). Promote or resolve the dependencies "
                        f"so {h.id} can auto-promote, or call "
                        f'abandon_hypothesis(id="{h.id}") before terminating.'
                    )
                else:
                    blockers.append(
                        f"{h.id} has VERIFIED review but was not promoted. "
                        f"This is unexpected — auto-promotion should have handled it. "
                        f'Abandon with abandon_hypothesis(id="{h.id}") before terminating.'
                    )
            else:
                blockers.append(
                    f"{h.id} has no VERIFIED review. "
                    f"Emit task_type: review targeting {h.id}, "
                    f'or call abandon_hypothesis(id="{h.id}"), before terminating.'
                )

    return (len(blockers) == 0, blockers)
