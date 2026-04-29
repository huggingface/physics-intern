"""Surveyor agent: produces background notes mapping the research landscape."""

from __future__ import annotations

from typing import TYPE_CHECKING

from open_dirac.llm import LLMResponse
from open_dirac.rendering import _problem_guidelines, render_background_survey_xml

from ..base import BaseAgent
from ..parsing import extract_json

# Text-valued section fields (sanity_checks is list[str], handled separately)
TEXT_SECTION_FIELDS = (
    "background",
    "key_insights",
    "known_methods",
    "known_pitfalls",
    "conventions_and_definitions",
    "expected_answer_structure",
    "problem_summary",
)

if TYPE_CHECKING:
    from open_dirac.config import Config
    from open_dirac.metrics import MetricsTracker
    from open_dirac.research_state import ResearchState
    from open_dirac.task import Task
    from open_dirac.workspace import WorkspaceManager


class SurveyorAgent(BaseAgent):
    name = "surveyor"
    prompt_file = "prompt.md"

    def __init__(
        self, config: Config, workspace: WorkspaceManager, metrics: MetricsTracker
    ):
        super().__init__(config, workspace, metrics)
        self.research_state: ResearchState | None = None
        self.parsed_survey: dict | None = None

    def _validate_response(self, response: LLMResponse) -> bool:
        """Check that at least one structured section was extracted from JSON."""
        text = (response.text or "").strip()
        parsed = extract_json(text)
        if isinstance(parsed, dict):
            return any(
                k in parsed and isinstance(parsed[k], str) and parsed[k].strip()
                for k in TEXT_SECTION_FIELDS
            )
        return False

    def _parse_retry_hint(self, parse_error: str | None = None) -> str:
        return (
            "Recall the required output format and provide it now:\n\n"
            "```json\n"
            "{\n"
            '  "background": "...",\n'
            '  "key_insights": "...",\n'
            '  "known_methods": "...",\n'
            '  "known_pitfalls": "...",\n'
            '  "conventions_and_definitions": "...",\n'
            '  "expected_answer_structure": "...",\n'
            '  "problem_summary": "...",\n'
            '  "sanity_checks": [\n'
            '    {"predicate": "...", "rationale": "..."},\n'
            '    {"predicate": "...", "rationale": "..."}\n'
            "  ]\n"
            "}\n"
            "```"
        )

    def build_context(self, task: Task, iteration: int) -> str:
        parts = [
            f"<problem-statement>\n"
            f"{self.research_state.problem_statement if self.research_state else ''}"
            f"\n</problem-statement>",
        ]
        if self.research_state and self.research_state.answer_template:
            parts.append(
                f"<answer-template>\n{self.research_state.answer_template}\n</answer-template>"
            )
        parts.append(_problem_guidelines())
        # On re-survey (iteration > 0), include current background survey + research state
        if iteration > 0 and self.research_state:
            survey_ctx = render_background_survey_xml(self.research_state)
            if survey_ctx:
                parts.append(
                    f"<current-background-survey>\n{survey_ctx}\n</current-background-survey>"
                )
            # Research state: conventions, established results, research questions, hypotheses
            rs_parts: list[str] = []
            if self.research_state.conventions:
                rs_parts.append(
                    f"<conventions>\n{self.research_state.conventions}\n</conventions>"
                )
            ers = self.research_state.established_hypotheses()
            if ers:
                er_lines = [f"- {h.id}: {h.statement}, VERIFIED" for h in ers]
                rs_parts.append(
                    "<established-results>\n"
                    + "\n".join(er_lines)
                    + "\n</established-results>"
                )
            rqs = [rq for rq in self.research_state.research_questions.values()]
            if rqs:
                rq_lines = [
                    f"- {rq.id}: {rq.question}, {rq.status.value.upper()}" for rq in rqs
                ]
                rs_parts.append(
                    "<research-questions>\n"
                    + "\n".join(rq_lines)
                    + "\n</research-questions>"
                )
            whs = self.research_state.working_hypotheses()
            if whs:
                wh_lines = []
                for h in whs:
                    verdict = (
                        h.review.verdict
                        if h.review
                        else ("PENDING REVIEW" if h.evidence else "no evidence yet")
                    )
                    wh_lines.append(f"- {h.id}: {h.statement}, {verdict}")
                rs_parts.append(
                    "<hypotheses>\n" + "\n".join(wh_lines) + "\n</hypotheses>"
                )
            if rs_parts:
                parts.append(
                    "<research-state>\n" + "\n".join(rs_parts) + "\n</research-state>"
                )
        return "\n\n".join(parts)

    def process_response(self, response: LLMResponse, task: Task, iteration: int):
        text = response.text.strip()
        sections: dict[str, str] = {}
        sanity_checks: list[dict[str, str]] = []

        parsed = extract_json(text)
        if isinstance(parsed, dict):
            sections = {
                k: parsed[k].strip()
                for k in TEXT_SECTION_FIELDS
                if k in parsed and isinstance(parsed[k], str) and parsed[k].strip()
            }
            raw_sc = parsed.get("sanity_checks")
            if isinstance(raw_sc, list):
                for item in raw_sc:
                    if isinstance(item, dict) and "predicate" in item:
                        sanity_checks.append(
                            {
                                "predicate": str(item["predicate"]).strip(),
                                "rationale": str(item.get("rationale", "")).strip(),
                            }
                        )
                    elif isinstance(item, dict) and "check" in item:
                        # Legacy format
                        sanity_checks.append({"predicate": str(item["check"]).strip()})
                    elif isinstance(item, str) and item.strip():
                        # Bare string fallback
                        sanity_checks.append({"predicate": item.strip()})
            elif isinstance(raw_sc, str) and raw_sc.strip():
                for line in raw_sc.splitlines():
                    line = line.lstrip("- ").strip()
                    if line:
                        sanity_checks.append({"predicate": line})

        self.parsed_survey = {
            "raw_notes": text,
            **sections,
            "sanity_checks": sanity_checks,
        }
