"""Adjudicator agent: neutral judge for critique adjudication."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from open_dirac.llm import LLMResponse
from open_dirac.rendering import render_research_context_xml

from ..base import BaseAgent
from ..parsing import JSON_FENCE_RE, try_json_loads

if TYPE_CHECKING:
    from open_dirac.research_state import ResearchState
    from open_dirac.task import Task


_BARE_ADJUDICATION_RE = re.compile(r'\{[^{}]*"adjudication"[^{}]*\}', re.DOTALL)


def _parse_adjudication_json(text: str) -> dict | None:
    """Extract the last JSON block containing an adjudication from model output."""
    fenced = list(JSON_FENCE_RE.finditer(text))
    if fenced:
        try:
            return try_json_loads(fenced[-1].group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass
    bare = list(_BARE_ADJUDICATION_RE.finditer(text))
    if bare:
        try:
            return try_json_loads(bare[-1].group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    return None


class AdjudicatorAgent(BaseAgent):
    name = "adjudicator"
    prompt_file = "prompt.md"
    tools = []  # one-shot: no tools

    def __init__(self, config, workspace, metrics):
        super().__init__(config, workspace, metrics)
        self.research_state: ResearchState | None = None
        self.adjudication_result: dict | None = None

    def _validate_response(self, response) -> bool:
        return _parse_adjudication_json(response.text or "") is not None

    def _parse_retry_hint(self, parse_error: str | None = None) -> str:
        return (
            "Recall the required output format and provide it now:\n\n"
            "```json\n"
            "{\n"
            '  "adjudication": "valid|invalid|needs_evidence",\n'
            '  "reasoning": "Detailed explanation of your ruling.",\n'
            '  "revised_verdict": "REFUTED (only if adjudication=valid)",\n'
            '  "counter_argument": "Why the critique fails (only if adjudication=invalid).",\n'
            '  "investigation_scope": "What evidence is needed (only if adjudication=needs_evidence)."\n'
            "}\n"
            "```"
        )

    def build_context(self, task: Task, iteration: int) -> str:
        """Build neutral adjudication context."""
        if not self.research_state:
            return ""

        parts: list[str] = []
        target_id = task.target_claim

        # 1. Research context — problem statement + answer template
        parts.append(render_research_context_xml(self.research_state))

        # 2. Background survey — known pitfalls only
        if self.research_state.known_pitfalls:
            parts.append(
                f"<background-survey>\n"
                f"<known-pitfalls>\n{self.research_state.known_pitfalls}\n</known-pitfalls>\n"
                f"</background-survey>"
            )

        # 3. Research state — conventions, established results, sanity checks
        rs_parts: list[str] = []
        if self.research_state.conventions:
            rs_parts.append(
                f"<conventions>\n{self.research_state.conventions}\n</conventions>"
            )
        ers = self.research_state.established_hypotheses()
        other_ers = [er for er in ers if er.id != target_id]
        if other_ers:
            er_lines = [f"- **{er.id}**: {er.statement}" for er in other_ers]
            rs_parts.append(
                "<established-results>\n"
                + "\n".join(er_lines)
                + "\n</established-results>"
            )
        if self.research_state.sanity_checks:
            checks_text = "\n".join(f"- {c}" for c in self.research_state.sanity_checks)
            rs_parts.append(f"<sanity-checks>\n{checks_text}\n</sanity-checks>")
        if rs_parts:
            parts.append(
                "<research-state>\n" + "\n".join(rs_parts) + "\n</research-state>"
            )

        # 4. Claim being challenged — full details
        if target_id and target_id in self.research_state.hypotheses:
            h = self.research_state.hypotheses[target_id]
            claim_parts: list[str] = [f"ID: {h.id}", f"Statement: {h.statement}"]
            if h.derivation:
                claim_parts.append(f"<derivation>\n{h.derivation}\n</derivation>")
            if h.evidence:
                for ev_idx, ev in enumerate(h.evidence, 1):
                    ev_parts: list[str] = []
                    if ev.approach:
                        ev_parts.append(f"<approach>\n{ev.approach}\n</approach>")
                    if ev.method:
                        ev_parts.append(f"<method>{ev.method}</method>")
                    if ev.result:
                        ev_parts.append(f"<result>{ev.result}</result>")
                    if ev.reasoning:
                        ev_parts.append(f"<reasoning>\n{ev.reasoning}\n</reasoning>")
                    if ev.confidence:
                        ev_parts.append(f"<confidence>{ev.confidence}</confidence>")
                    if ev.scripts:
                        for script_name in ev.scripts:
                            purpose = ev.script_purposes.get(script_name, "")
                            try:
                                code = self.workspace.read_file(
                                    f"computations/{script_name}"
                                )
                            except Exception:
                                code = "[not found]"
                            stem = Path(script_name).stem
                            try:
                                output = self.workspace.read_file(
                                    f"computations/{stem}.output"
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
                            content = self.workspace.read_file(
                                f"derivations/{ev.derivation_file}"
                            )
                        except Exception:
                            content = ""
                        ev_parts.append(
                            f'<derivation file="{ev.derivation_file}">\n'
                            f"{content or ev.reasoning}\n</derivation>"
                        )
                    label = (
                        f' n="{ev_idx}/{len(h.evidence)}"'
                        if len(h.evidence) > 1
                        else ""
                    )
                    ev_parts_str = "\n".join(ev_parts)
                    claim_parts.append(
                        f'<evidence type="{ev.type}"{label}>\n{ev_parts_str}\n</evidence>'
                    )
            if h.review:
                claim_parts.append(f"Original review verdict: {h.review.verdict}")
                if h.review.summary:
                    claim_parts.append(f"Original review summary: {h.review.summary}")
            parts.append(
                f'<claim id="{target_id}">\n' + "\n".join(claim_parts) + "\n</claim>"
            )

        # 5. Challenge (critique argument)
        if task.critique_argument:
            parts.append(f"<challenge>\n{task.critique_argument}\n</challenge>")

        return "\n\n".join(parts)

    def process_response(self, response: LLMResponse, task: Task, iteration: int):
        """Parse adjudication JSON from one-shot response."""
        text = response.text or ""
        parsed = _parse_adjudication_json(text)

        if parsed and "adjudication" in parsed:
            adjudication = parsed["adjudication"]
            if adjudication not in ("valid", "invalid", "needs_evidence"):
                adjudication = "needs_evidence"  # safe fallback
            self.adjudication_result = {
                "adjudication": adjudication,
                "reasoning": parsed.get("reasoning", ""),
                "revised_verdict": parsed.get("revised_verdict", ""),
                "counter_argument": parsed.get("counter_argument", ""),
                "investigation_scope": parsed.get("investigation_scope", ""),
            }
        else:
            # Parse failure — default to needs_evidence (safest)
            self.adjudication_result = {
                "adjudication": "needs_evidence",
                "reasoning": "Failed to parse adjudication output.",
                "investigation_scope": "Re-run adjudication with clearer output format.",
            }
