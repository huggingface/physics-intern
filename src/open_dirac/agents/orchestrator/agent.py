"""Orchestrator agent: plans and coordinates research via state-mutation tools."""

from __future__ import annotations

from collections.abc import Callable

from open_dirac.llm import AgentResult, LLMResponse, run_agent_loop
from open_dirac.research_state import ResearchState
from open_dirac.rendering import (
    render_background_survey_xml,
    render_research_context_xml,
)
from open_dirac.task import Task, TaskType, TASK_TYPE_AGENT_MAP
from open_dirac.tool_call import ToolCall

from ..base import BaseAgent
from .context import render_orchestrator_slim_state
from .tools import OrchestratorToolExecutor


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"
    prompt_file = "prompt.md"
    tools = OrchestratorToolExecutor.TOOL_DEFINITIONS

    def __init__(self, config, workspace, metrics):
        super().__init__(config, workspace, metrics)
        self.context_suffix: str = ""
        self.dispatch_history_text: str = ""
        self._tool_executor: OrchestratorToolExecutor | None = None
        self.research_state: ResearchState | None = None

    def build_context(self, task: Task, iteration: int) -> str:
        parts: list[str] = []

        # 1. Research context — problem statement + answer template
        if self.research_state:
            parts.append(render_research_context_xml(self.research_state))

        # 2. Background survey — survey data with known pitfalls
        if self.research_state:
            survey_ctx = render_background_survey_xml(self.research_state)
            if survey_ctx:
                parts.append(f"<background-survey>\n{survey_ctx}\n</background-survey>")

        # 3. Conventions reminder
        if (
            iteration >= 3
            and self.research_state
            and not self.research_state.conventions
        ):
            parts.append(
                ">>> REMINDER: The '# Conventions' section is still empty. "
                "Consider populating it with the unit system, sign conventions, "
                "and variable definitions being used. <<<"
            )

        # 4. Research state — conventions, strategy, entities, research notes, dispatch history
        state_text = (
            render_orchestrator_slim_state(
                self.research_state,
                max_open_rqs=self.config.max_open_rqs,
            )
            if self.research_state
            else ""
        )
        rs_inner_parts: list[str] = []
        if state_text:
            rs_inner_parts.append(state_text)
        if self.research_state and self.research_state.research_notes:
            cutoff = max(iteration - 4, 0)
            recent = [
                n
                for n in self.research_state.research_notes
                if n.get("iteration", 0) >= cutoff
            ]
            if recent:
                note_lines = []
                for note in recent:
                    note_lines.append(
                        f"- [iter {note.get('iteration', '?')}] {note.get('text', '')}"
                    )
                rs_inner_parts.append(
                    "<research-notes>\n" + "\n".join(note_lines) + "\n</research-notes>"
                )
        if self.dispatch_history_text:
            rs_inner_parts.append(self.dispatch_history_text)
            self.dispatch_history_text = ""
        if rs_inner_parts:
            parts.append(
                "<research-state>\n"
                + "\n\n".join(rs_inner_parts)
                + "\n</research-state>"
            )

        # 5. Inter-iteration banners (evidence results, verified hypotheses, etc.)
        if self.context_suffix:
            parts.append(self.context_suffix)
            self.context_suffix = ""
        return "\n\n".join(parts)

    def _call_with_tools(
        self,
        context: str,
        task: Task,
        iteration: int,
        on_round: Callable[[int, str, list[ToolCall], int, int, int, int, float], None]
        | None = None,
    ) -> AgentResult:
        """Run the orchestrator with state-mutation tools."""
        self._tool_executor = OrchestratorToolExecutor(
            workspace=self.workspace,
            iteration=iteration,
            research_state=self.research_state,
            min_er_for_completion=self.config.min_er_for_completion,
            max_iterations=self.config.max_iterations,
            budget_synthesis_margin=self.config.budget_synthesis_margin,
            max_open_rqs=self.config.max_open_rqs,
            rq_evidence_cap=self.config.rq_evidence_cap,
            max_refuted_retries=self.config.max_refuted_retries,
        )
        result = run_agent_loop(
            system=self.system_prompt,
            user_content=context,
            config=self.config,
            tool_executor=self._tool_executor,
            tools=self.tools,
            max_rounds=self.config.max_tool_rounds,
            agent_name=self.name,
            iteration=iteration,
            on_round=on_round,
        )

        self.metrics.record_call(
            iteration=iteration,
            agent=self.name,
            input_tokens=result.total_input_tokens,
            output_tokens=result.total_output_tokens,
            duration=result.duration,
            max_tokens_hit=result.truncated,
            rounds=result.rounds,
            tool_calls=len(result.tool_calls),
            truncated=result.truncated,
            reasoning_tokens=result.total_reasoning_tokens,
            answer_tokens=result.total_answer_tokens,
        )
        return result

    def process_response(
        self, response: LLMResponse | AgentResult, task: Task, iteration: int
    ):
        """Process orchestrator output — state already mutated by tool executor."""
        if not self._tool_executor:
            return

        # Always write CURRENT_TASK.md when a task was dispatched, even if no
        # state mutations occurred.  Without this, verify agents read a stale
        # file and target the wrong hypothesis.
        if self._tool_executor.task_data:
            task_obj = self._task_from_tool_data(
                self._tool_executor.task_data, iteration
            )
            self.workspace.write_file("CURRENT_TASK.md", task_obj.to_markdown())

    def _task_from_tool_data(self, data: dict, iteration: int) -> Task:
        """Build a Task from dispatch tool arguments."""

        def _str(val: object, default: str = "") -> str:
            """Coerce to str — LLMs sometimes return a list instead of a string."""
            if val is None:
                return default
            if isinstance(val, list):
                return "\n".join(str(v) for v in val)
            return str(val)

        raw_type = data.get("task_type", "research")
        try:
            task_type = TaskType(raw_type)
        except ValueError:
            task_type = TaskType.RESEARCH
        assigned_to = TASK_TYPE_AGENT_MAP.get(task_type, "researcher")
        return Task(
            task_id=f"TASK-{iteration:03d}",
            task_type=task_type,
            assigned_to=assigned_to,
            priority=data.get("priority", "medium"),
            iteration=iteration,
            target_claim=_str(data.get("target_claim")),
            body=_str(data.get("description")),
            background=_str(data.get("background")),
            method_hints=data.get("method_hints", []),
            assumptions=data.get("assumptions", []),
            relevant_results=data.get("relevant_results", []),
            recommended_sanity_checks=data.get("recommended_sanity_checks", []),
            answer_ers=data.get("answer_ers", []),
        )

    def parse_task(self, text: str, iteration: int = 0) -> Task:
        """Parse task from tool executor."""
        if self._tool_executor and self._tool_executor.task_data:
            return self._task_from_tool_data(self._tool_executor.task_data, iteration)
        # Fallback: parse from text frontmatter
        return Task.from_frontmatter(text, fallback_iteration=iteration)
