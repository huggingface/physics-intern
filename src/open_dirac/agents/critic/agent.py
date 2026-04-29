"""Deep Critic agent: one-shot structured JSON review."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from open_dirac.console import console
from open_dirac.llm import LLMResponse
from open_dirac.research_state import Critique, CritiqueStatus, Severity
from open_dirac.utils.categories import CompensationCategory as CC
from open_dirac.workspace import log_scaffold_event

from ..base import BaseAgent
from ..parsing import JSON_FENCE_RE, try_json_loads
from .context import render_critic_context

if TYPE_CHECKING:
    from open_dirac.research_state import ResearchState
    from open_dirac.task import Task


# ---------------------------------------------------------------------------
# JSON parsing (handles nested critiques array)
# ---------------------------------------------------------------------------


def _parse_critic_json(text: str) -> dict | None:
    """Extract the last JSON block containing a critiques array from model output.

    Tries fenced ```json blocks first (last match), then falls back to
    brace-counting for bare JSON with nested objects.
    """
    # Prefer fenced ```json blocks — take the last one
    fenced = list(JSON_FENCE_RE.finditer(text))
    if fenced:
        try:
            parsed = try_json_loads(fenced[-1].group(1).strip())
            if isinstance(parsed, dict) and "critiques" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Bare JSON fallback: find outermost { ... } containing "critiques"
    # using brace-counting (needed because critiques array has nested objects)
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                if '"critiques"' in candidate:
                    try:
                        parsed = try_json_loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
                start = None

    return None


class CriticAgent(BaseAgent):
    name = "deep_critic"
    prompt_file = "prompt.md"
    tools = []  # one-shot: no tools

    def _validate_response(self, response: LLMResponse) -> bool:
        return _parse_critic_json(response.text or "") is not None

    def _parse_retry_hint(self, parse_error: str | None = None) -> str:
        return (
            "Recall the required output format and provide it now:\n\n"
            "```json\n"
            "{\n"
            '  "summary": "Concise audit trail.",\n'
            '  "critiques": [\n'
            "    {\n"
            '      "target_id": "STRATEGY or WH-NNN or ER-NNN",\n'
            '      "target_type": "er|strategy|coordination",\n'
            '      "severity": "HIGH|MEDIUM|LOW",\n'
            '      "argument": "What is wrong and why it matters."\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```"
        )

    def __init__(self, config, workspace, metrics):
        super().__init__(config, workspace, metrics)
        self._no_critiques_filed: bool = False
        self.research_state: ResearchState | None = None

    def build_context(self, task: Task, iteration: int) -> str:
        return (
            render_critic_context(self.research_state, iteration)
            if self.research_state
            else ""
        )

    def process_response(self, response: LLMResponse, task: Task, iteration: int):
        """Parse structured JSON from one-shot response text."""
        text = response.text or ""
        parsed = _parse_critic_json(text)

        if parsed is None:
            # Parse failure — treat as clean review
            self._no_critiques_filed = True
            log_scaffold_event(
                self.workspace.root,
                iteration,
                CC.OUTPUT_NORMALIZATION,
                "critic_json_parse_failure",
                f"text_length={len(text)}",
            )
            if self.research_state:
                self.research_state.critic_clean_reviews.append(
                    {
                        "iteration": iteration,
                        "summary": "Parse failure — no critiques extracted.",
                    }
                )
            return

        critiques_data = parsed.get("critiques", [])
        summary = parsed.get("summary", "")

        if not critiques_data:
            # Clean review — no issues found
            self._no_critiques_filed = True
            log_scaffold_event(
                self.workspace.root,
                iteration,
                CC.LOOP_CONTROL,
                "no_critiques_filed",
                f"summary={summary}",
            )
            if self.research_state:
                self.research_state.critic_clean_reviews.append(
                    {
                        "iteration": iteration,
                        "summary": summary,
                    }
                )
            return

        # Critiques present — assign IDs and store
        self._no_critiques_filed = False
        if not self.research_state:
            return

        for crit_data in critiques_data:
            crit_num = self.research_state.next_critique_num()
            crit_id = f"CRIT-{crit_num:03d}"

            # Validate severity, default to MEDIUM
            raw_severity = crit_data.get("severity", "MEDIUM")
            try:
                sev = Severity(raw_severity)
            except ValueError:
                sev = Severity.MEDIUM

            target_id = crit_data.get("target_id", "")
            target_type = crit_data.get("target_type", "")
            argument = crit_data.get("argument", "")

            crit = Critique(
                id=crit_id,
                targets=[target_id] if target_id else [],
                severity=sev,
                argument=argument,
                status=CritiqueStatus.ACTIVE,
                target_type=target_type,
                iteration_filed=iteration,
            )
            self.research_state.critiques[crit.id] = crit

            # Console output + scaffold event
            sev_label = sev.value
            target_str = target_id or "general"
            arg_short = argument[:80]
            console.print(
                f"  [yellow]{crit.id}[/] [{sev_label}] targeting {target_str}: {arg_short}"
            )
            log_scaffold_event(
                self.workspace.root,
                iteration,
                CC.STATE_INVARIANTS,
                "file_critique",
                f"{crit.id} [{sev_label}] → {target_str}: {argument[:120]}",
            )

            # Link to hypothesis
            for t in crit.targets:
                if t in self.research_state.hypotheses:
                    h = self.research_state.hypotheses[t]
                    if crit.id not in h.critiques:
                        h.critiques.append(crit.id)
