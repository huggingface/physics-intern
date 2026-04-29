"""Reviewer agent: one-shot structured review of hypotheses."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from open_dirac.llm import LLMResponse, ParseFailureError
from open_dirac.rendering import _render_sanity_checks, render_research_context_xml
from open_dirac.research_state import ReviewResult

from ..base import BaseAgent
from ..parsing import JSON_FENCE_RE, try_json_loads

if TYPE_CHECKING:
    from open_dirac.research_state import ResearchState
    from open_dirac.task import Task


# Match a bare top-level { ... } object containing "verdict"
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"verdict\"[^{}]*\}", re.DOTALL)


def _parse_review_json(text: str) -> dict | None:
    """Extract the last JSON block containing a verdict from model output."""
    # Prefer fenced ```json blocks — take the last one
    fenced = list(JSON_FENCE_RE.finditer(text))
    if fenced:
        try:
            return try_json_loads(fenced[-1].group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass
    # Fall back to bare JSON objects containing "verdict"
    bare = list(_BARE_JSON_RE.finditer(text))
    if bare:
        try:
            return try_json_loads(bare[-1].group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    return None


_CODE_CHAR_LIMIT = 5_000
_OUTPUT_CHAR_LIMIT = 10_000


def _truncate(text: str, limit: int) -> str:
    """Truncate *text* to *limit* chars, preserving head and tail."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n\n[... truncated {len(text) - limit} chars ...]\n\n"
        + text[-half:]
    )


class ReviewerAgent(BaseAgent):
    name = "reviewer"
    prompt_file = "prompt.md"
    tools = []  # one-shot: no tools
    raise_on_parse_failure = True

    def _validate_response(self, response: LLMResponse) -> bool:
        return _parse_review_json(response.text or "") is not None

    def _parse_retry_hint(self, parse_error: str | None = None) -> str:
        return (
            "Recall the required output format and provide it now:\n\n"
            "```json\n"
            "{\n"
            '  "verdict": "VERIFIED|REFUTED|INCONCLUSIVE",\n'
            '  "summary": "1-3 sentence summary of the review outcome.",\n'
            '  "details": "Detailed reasoning for your verdict.",\n'
            '  "sanity_checks": [\n'
            '    {"check": "...", "type": "constraint|conjecture", '
            '"outcome": "PASS|FAIL|N/A", "reasoning": "..."}\n'
            "  ]\n"
            "}\n"
            "```"
        )

    def __init__(self, config, workspace, metrics):
        super().__init__(config, workspace, metrics)
        self.research_state: ResearchState | None = None

    def _auto_review_description(self, target_id: str) -> str:
        """Generate a review task description from the WH and its evidence."""
        h = (
            self.research_state.hypotheses.get(target_id)
            if self.research_state
            else None
        )
        if not h:
            return f"Review {target_id}."

        ev_types = (
            sorted(set(ev.type for ev in h.evidence)) if h.evidence else ["research"]
        )
        ev_type = "+".join(ev_types) if len(ev_types) > 1 else ev_types[0]

        # Find originating RQ
        rq_context = ""
        if self.research_state:
            for rq in self.research_state.research_questions.values():
                if target_id in rq.resolved_to:
                    rq_context = f"\nOriginating question ({rq.id}): {rq.question}"
                    if rq.context:
                        rq_context += f"\nContext: {rq.context}"
                    break

        if ev_type == "compute":
            desc = (
                f"Review {target_id}: audit the computational approach and code, checking "
                "implementation correctness, result interpretation, and physical consistency "
                "with expected qualitative behavior."
            )
        else:
            desc = (
                f"Review {target_id}: verify the analytical derivation, checking mathematical "
                "correctness, convention consistency with the problem's symbol definitions, "
                "and physical sanity (expected scaling, limiting cases, dimensional analysis)."
            )

        if rq_context:
            desc += rq_context

        return desc

    def build_context(self, task: Task, iteration: int) -> str:
        """Build focused verification context: WH + evidence + light state."""
        parts: list[str] = []

        # 1. Research context — problem statement + answer template
        if self.research_state:
            parts.append(render_research_context_xml(self.research_state))

        # 2. Background survey — known pitfalls only
        if self.research_state and self.research_state.known_pitfalls:
            parts.append(
                f"<background-survey>\n"
                f"<known-pitfalls>\n{self.research_state.known_pitfalls}\n</known-pitfalls>\n"
                f"</background-survey>"
            )

        # 3. Research state — conventions, established results, sanity checks
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
            if self.research_state.sanity_checks:
                rs_parts.append(
                    _render_sanity_checks(
                        self.research_state.sanity_checks,
                        tag="suggested-sanity-checks",
                    )
                )
            if rs_parts:
                parts.append(
                    "<research-state>\n" + "\n".join(rs_parts) + "\n</research-state>"
                )

        # 4. Original question (before claim, for context)
        if self.research_state and task.target_claim:
            target_id = task.target_claim
            for rq in self.research_state.research_questions.values():
                if target_id in rq.resolved_to:
                    rq_content = f"{rq.id}: {rq.question}"
                    parts.append(
                        f'<original-question id="{rq.id}">\n{rq_content}\n</original-question>'
                    )
                    break

        # 5. Claim + evidence
        if self.research_state and task.target_claim:
            target_id = task.target_claim
            h = self.research_state.hypotheses.get(target_id)
            if h:
                claim_parts: list[str] = [f"Statement: {h.statement}"]
                if h.derivation:
                    claim_parts.append(f"<derivation>\n{h.derivation}\n</derivation>")
                parts.append(
                    f'<claim id="{target_id}">\n'
                    + "\n".join(claim_parts)
                    + "\n</claim>"
                )

                # Evidence (iterate over all items)
                if h.evidence:
                    multi = len(h.evidence) > 1
                    for ev_idx, ev in enumerate(h.evidence, 1):
                        ev_parts: list[str] = []
                        if ev.description:
                            ev_parts.append(
                                f"<description>{ev.description}</description>"
                            )
                        if ev.summary:
                            ev_parts.append(f"<summary>{ev.summary}</summary>")
                        if ev.approach:
                            ev_parts.append(f"<approach>\n{ev.approach}\n</approach>")
                        if ev.method:
                            ev_parts.append(f"<method>{ev.method}</method>")
                        if ev.result:
                            ev_parts.append(f"<result>{ev.result}</result>")
                        if ev.scripts:
                            for script_name in ev.scripts:
                                purpose = ev.script_purposes.get(script_name, "")
                                try:
                                    code = _truncate(
                                        self.workspace.read_file(
                                            f"computations/{script_name}"
                                        ),
                                        _CODE_CHAR_LIMIT,
                                    )
                                except Exception:
                                    code = "[not found]"
                                stem = Path(script_name).stem
                                try:
                                    output = _truncate(
                                        self.workspace.read_file(
                                            f"computations/{stem}.output"
                                        ),
                                        _OUTPUT_CHAR_LIMIT,
                                    )
                                except Exception:
                                    output = "[not found]"
                                comp_parts = []
                                if purpose:
                                    comp_parts.append(f"  <purpose>{purpose}</purpose>")
                                comp_parts.append(
                                    f'  <code language="python">\n{code}\n  </code>'
                                )
                                comp_parts.append(f"  <output>\n{output}\n  </output>")
                                ev_parts.append(
                                    f'<computation name="{script_name}">\n'
                                    + "\n".join(comp_parts)
                                    + "\n</computation>"
                                )
                        if ev.derivation_file:
                            try:
                                content = _truncate(
                                    self.workspace.read_file(
                                        f"derivations/{ev.derivation_file}"
                                    ),
                                    _OUTPUT_CHAR_LIMIT,
                                )
                            except Exception:
                                content = ""
                            ev_parts.append(
                                f'<derivation file="{ev.derivation_file}">\n'
                                f"{content or ev.reasoning}\n</derivation>"
                            )
                        elif ev.reasoning:
                            ev_parts.append(
                                f"<reasoning>\n{ev.reasoning}\n</reasoning>"
                            )
                        if ev.notes:
                            ev_parts.append(f"<notes>{ev.notes}</notes>")
                        if ev.confidence:
                            ev_parts.append(f"<confidence>{ev.confidence}</confidence>")
                        label = f' n="{ev_idx}/{len(h.evidence)}"' if multi else ""
                        parts.append(
                            f'<evidence type="{ev.type}"{label}>\n'
                            + "\n".join(ev_parts)
                            + "\n</evidence>"
                        )

        # 6. Instructions (task description, at the end)
        if self.research_state and task.target_claim:
            auto_desc = self._auto_review_description(task.target_claim)
        else:
            auto_desc = task.render_agent_context(include_structured=False)
        parts.append(f"<instructions>\n{auto_desc}\n</instructions>")

        return "\n\n".join(parts)

    def process_response(self, response: LLMResponse, task: Task, iteration: int):
        """Parse structured JSON verdict from one-shot response text."""
        text = response.text or ""
        parsed = _parse_review_json(text)

        if parsed and "verdict" in parsed:
            verdict = parsed["verdict"]
            if verdict not in ("VERIFIED", "REFUTED", "INCONCLUSIVE"):
                verdict = "INCONCLUSIVE"
            review = ReviewResult(
                verdict=verdict,
                summary=parsed.get("summary", ""),
                details=parsed.get("details", ""),
                iteration=iteration,
            )
        else:
            # Unreachable when raise_on_parse_failure=True — _call_with_retry
            # raises ParseFailureError before process_response is called.
            raise ParseFailureError(
                agent_name=self.name,
                detail="process_response reached without valid parsed output",
            )

        # Store on target hypothesis
        if self.research_state:
            target_id = task.target_claim
            if target_id and target_id in self.research_state.hypotheses:
                self.research_state.hypotheses[target_id].review = review
