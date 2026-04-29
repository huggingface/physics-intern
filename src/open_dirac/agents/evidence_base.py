"""Shared base for evidence-producing agents (researcher, computer).

Provides common context building, target resolution, evidence storage,
and relevant-results rendering. Subclasses only need to define
`name`, `prompt_file`, `tools`, and `process_response`.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from typing import TYPE_CHECKING

from ..rendering import render_research_context_xml
from ..research_state import Evidence
from .base import BaseAgent

if TYPE_CHECKING:
    from ..llm import AgentResult, LLMResponse
    from ..research_state import ResearchState
    from ..task import Task

ENTITY_ID_RE = re.compile(r"(?:ER|WH|RQ)-\d+")
_RELEVANT_ID_RE = re.compile(r"^(?:ER|WH|RQ)-\d+$")


def render_relevant_results(
    refs: list[str],
    research_state: "ResearchState | None",
) -> str:
    """Resolve relevant_results IDs to their content for agent context."""
    lines: list[str] = []
    for ref in refs:
        ref = ref.strip()
        if not research_state or not _RELEVANT_ID_RE.match(ref):
            lines.append(f"- {ref}")
            continue

        # Resolve entity
        entity_line = f"- **{ref}**"
        ev_summary = ""
        if ref in research_state.hypotheses:
            h = research_state.hypotheses[ref]
            entity_line += f": {h.statement}" if h.statement else ""
            if h.evidence:
                ev = h.evidence[-1]  # most recent
                parts = []
                if ev.summary:
                    parts.append(ev.summary)
                elif ev.result:
                    parts.append(ev.result[:200])
                if ev.confidence:
                    parts.append(ev.confidence)
                if parts:
                    ev_summary = f"  Evidence: {' — '.join(parts)}"
        elif ref in research_state.research_questions:
            rq = research_state.research_questions[ref]
            entity_line += f": {rq.question}" if rq.question else ""
            if rq.evidence:
                ev = rq.evidence[-1]  # most recent
                parts = []
                if ev.summary:
                    parts.append(ev.summary)
                elif ev.result:
                    parts.append(ev.result[:200])
                if ev.confidence:
                    parts.append(ev.confidence)
                if parts:
                    ev_summary = f"  Evidence: {' — '.join(parts)}"
        else:
            entity_line += " (not found in current state)"

        lines.append(entity_line)
        if ev_summary:
            lines.append(ev_summary)
    return "\n".join(lines)


class EvidenceAgent(BaseAgent):
    """Base class for researcher and computer agents.

    Provides shared context building (background → target → task → research context),
    target resolution, and evidence storage. Subclasses implement process_response.
    """

    def __init__(self, config, workspace, metrics):
        super().__init__(config, workspace, metrics)
        self.research_state: ResearchState | None = None

    def build_context(self, task: Task, iteration: int) -> str:
        parts: list[str] = []

        # 1. Research context — problem statement + answer template
        if self.research_state:
            parts.append(render_research_context_xml(self.research_state))

        # 2. Research state — conventions, established results
        if self.research_state:
            rs_parts: list[str] = []
            if self.research_state.conventions:
                rs_parts.append(
                    f"<conventions>\n{self.research_state.conventions}\n</conventions>"
                )
            ers = self.research_state.established_hypotheses()
            if ers:
                er_lines = [f"- **{er.id}**: {er.statement}" for er in ers]
                rs_parts.append(
                    "<established-results>\n"
                    + "\n".join(er_lines)
                    + "\n</established-results>"
                )
            if rs_parts:
                parts.append(
                    "<research-state>\n" + "\n".join(rs_parts) + "\n</research-state>"
                )

        # 3. Task — target, background, instructions, method hints, assumptions, relevant results
        task_parts: list[str] = []
        target_text = self._resolve_target_text(task.target_claim)
        if target_text:
            task_parts.append(
                f"<target>\n{task.target_claim}: {target_text}\n</target>"
            )
        if task.background:
            task_parts.append(f"<background>\n{task.background}\n</background>")
        if task.body:
            task_parts.append(f"<instructions>\n{task.body}\n</instructions>")
        if task.method_hints:
            hints = "\n".join(f"- {h}" for h in task.method_hints)
            task_parts.append(f"<method-hints>\n{hints}\n</method-hints>")
        if task.assumptions:
            items = "\n".join(f"- {a}" for a in task.assumptions)
            task_parts.append(f"<assumptions>\n{items}\n</assumptions>")
        if task.relevant_results:
            items = render_relevant_results(task.relevant_results, self.research_state)
            task_parts.append(f"<relevant-results>\n{items}\n</relevant-results>")
        if task.recommended_sanity_checks:
            items = "\n".join(f"- {c}" for c in task.recommended_sanity_checks)
            task_parts.append(
                f"<recommended-sanity-checks>\n{items}\n</recommended-sanity-checks>"
            )
        if task_parts:
            parts.append("<task>\n" + "\n\n".join(task_parts) + "\n</task>")

        return "\n\n".join(parts)

    def _resolve_target_text(self, target_claim: str) -> str | None:
        """Resolve a target_claim ID to its question or statement text."""
        if not self.research_state or not target_claim:
            return None
        if target_claim in self.research_state.research_questions:
            rq = self.research_state.research_questions[target_claim]
            if rq.context:
                return f"{rq.question}\nContext: {rq.context}"
            return rq.question
        if target_claim in self.research_state.hypotheses:
            return self.research_state.hypotheses[target_claim].statement
        if target_claim in self.research_state.critiques:
            c = self.research_state.critiques[target_claim]
            text = f"[{c.severity.value}] {c.argument}"
            if c.targets:
                text += f"\nTargets: {', '.join(c.targets)}"
                # Include targeted entity content for better context
                for tid in c.targets:
                    entity_text = self._resolve_entity_text(tid)
                    if entity_text:
                        text += f"\n\n{tid}: {entity_text}"
            return text
        return None

    def _resolve_entity_text(self, entity_id: str) -> str | None:
        """Resolve an entity ID to its statement/question text (non-recursive)."""
        if not self.research_state:
            return None
        if entity_id in self.research_state.research_questions:
            return self.research_state.research_questions[entity_id].question
        if entity_id in self.research_state.hypotheses:
            return self.research_state.hypotheses[entity_id].statement
        return None

    def _render_active_critiques(self, task: "Task") -> str:
        """Render active critiques targeting the task's entity."""
        if not self.research_state:
            return ""
        target = task.target_claim
        seen_ids: set[str] = set()
        lines: list[str] = []
        # 1. Active critiques filed against the target entity
        if target:
            for c in self.research_state.active_critiques_for(target):
                if c.id not in seen_ids:
                    lines.append(f"- **{c.id}** [{c.severity.value}]: {c.argument}")
                    seen_ids.add(c.id)
        # 2. Explicit blocking_critiques (from CRIT redirect)
        for crit_id in task.blocking_critiques:
            if crit_id not in seen_ids:
                crit = self.research_state.critiques.get(crit_id)
                if crit:
                    lines.append(
                        f"- **{crit.id}** [{crit.severity.value}]: {crit.argument}"
                    )
                    seen_ids.add(crit.id)
        if not lines:
            return ""
        return "<active-critiques>\n" + "\n".join(lines) + "\n</active-critiques>"

    def _store_evidence(self, target_id: str, evidence: Evidence):
        """Store evidence on the target entity (RQ, WH, or critique).

        Clears any previously-refuted evidence before appending,
        so contradictory old results don't coexist with new ones.
        """
        state = self.research_state
        if not state:
            return

        if not evidence.id:
            evidence.id = f"EV-{state.next_evidence_num():03d}"

        def _append(ev_list: list[Evidence]):
            ev_list[:] = [ev for ev in ev_list if not ev.refuted]
            ev_list.append(evidence)

        if target_id in state.research_questions:
            _append(state.research_questions[target_id].evidence)
        elif target_id in state.hypotheses:
            _append(state.hypotheses[target_id].evidence)
        elif target_id in state.critiques:
            _append(state.critiques[target_id].evidence)

    @abstractmethod
    def process_response(
        self, response: "LLMResponse | AgentResult", task: "Task", iteration: int
    ): ...
