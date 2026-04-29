"""Formatter agent: produces clean ANSWER.md from final research state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from open_dirac.llm import LLMResponse

from ..base import BaseAgent
from .context import render_formatter_context

if TYPE_CHECKING:
    from open_dirac.config import Config
    from open_dirac.metrics import MetricsTracker
    from open_dirac.research_state import ResearchState
    from open_dirac.task import Task
    from open_dirac.workspace import WorkspaceManager

_REJECTION_PREFIX = "FORMATTER_REJECTION:"

_BEST_EFFORT_PREAMBLE = """\
<best-effort-mode>
The iteration limit was reached before the research loop completed normally.
Produce the BEST POSSIBLE answer from whatever results are available.

- Use Established Results (ERs) where they exist.
- Where ERs are missing, use the best available Working Hypothesis result \
from the <unverified-results> section. Clearly note which parts come from \
unverified results.
- Do NOT emit a FORMATTER_REJECTION — always produce a completed answer, \
even if some values are approximate or unverified.
- If an answer template is provided, fill in every placeholder with the best \
available value. Use comments to flag unverified placeholders.
</best-effort-mode>

"""


class FormatterAgent(BaseAgent):
    name = "formatter"
    prompt_file = "prompt.md"

    def __init__(
        self,
        config: Config,
        workspace: WorkspaceManager,
        metrics: MetricsTracker,
        answer_template: str = "",
    ):
        super().__init__(config, workspace, metrics)
        self.answer_template = answer_template
        self.research_state: ResearchState | None = None
        self.rejection_reason: str | None = None
        self.best_effort: bool = False

    def build_context(self, task: Task, iteration: int) -> str:
        # If instance-level template override exists, temporarily set on research_state
        # so the renderer emits it (renderer reads state.answer_template)
        if (
            self.answer_template
            and self.research_state
            and not self.research_state.answer_template
        ):
            self.research_state.answer_template = self.answer_template
        ctx = render_formatter_context(
            self.research_state,
            answer_ers=task.answer_ers,
            best_effort=self.best_effort,
        )
        if self.best_effort:
            ctx = _BEST_EFFORT_PREAMBLE + ctx
        return ctx

    def process_response(self, response: LLMResponse, task: Task, iteration: int):
        """Write ANSWER.md, checking for LLM-emitted rejection marker."""
        self.rejection_reason = None
        text = response.text or ""

        # Check for LLM-emitted rejection marker
        if text.lstrip().startswith(_REJECTION_PREFIX):
            self.rejection_reason = (
                text.lstrip().split("\n", 1)[0].removeprefix(_REJECTION_PREFIX).strip()
            )

        # Always write — circuit breaker may accept best-effort
        self.workspace.write_file("ANSWER.md", text)
