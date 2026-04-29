"""Resume helpers.

``reconstruct_loop_state`` rebuilds the durable fields of ``LoopState``
from a loaded ``ResearchState``. ``find_last_critic_iteration`` parses
``EVENT_LOG.jsonl`` for the last deep-critic LLM call so the critic
cadence survives a resume. Both are pure helpers with no engine
dependency; ``OpenDirac.resume`` wires them in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .loop_state import LoopState
from .research_state import Verdict

if TYPE_CHECKING:
    from .research_state import ResearchState


def reconstruct_loop_state(research_state: ResearchState) -> LoopState:
    """Rebuild LoopState from a loaded ResearchState.

    Only reconstructs durable fields — consumed-once banners are always
    empty between iterations, so they default to empty.
    """
    state = LoopState()

    # claim_failure_count: hypotheses with non-VERIFIED review that are still WORKING
    for h in research_state.hypotheses.values():
        if (
            h.review
            and h.review.verdict in (Verdict.REFUTED, "INCONCLUSIVE")
            and h.status == "working"
        ):
            state.claim_failure_count[h.id] = 1

    # last_content_iteration: max iteration from evidence/review across entities
    max_iter = 0
    for h in research_state.hypotheses.values():
        for ev in h.evidence:
            if ev.iteration is not None:
                max_iter = max(max_iter, ev.iteration)
        if h.review and h.review.iteration is not None:
            max_iter = max(max_iter, h.review.iteration)
    for rq in research_state.research_questions.values():
        for ev in rq.evidence:
            if ev.iteration is not None:
                max_iter = max(max_iter, ev.iteration)
    state.last_content_iteration = max_iter

    return state


def find_last_critic_iteration(workspace_path: Path | str) -> int:
    """Parse EVENT_LOG.jsonl for the last deep_critic LLM call iteration."""
    log_path = Path(workspace_path) / "EVENT_LOG.jsonl"
    if not log_path.exists():
        return 0
    max_iter = 0
    try:
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("kind") == "llm_call" and entry.get("agent") == "deep_critic":
                max_iter = max(max_iter, entry.get("iter", 0))
    except OSError:
        return 0
    return max_iter
