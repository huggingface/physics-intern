"""Agent dispatch and error handling.

Routes tasks to the right agent based on ``TaskType`` and handles the
recoverable failure modes (context too long, parse failure, transient
dispatch errors, max-tokens / max-rounds truncation). All functions take
explicit ``loop_state`` / ``workspace`` / ``metrics`` arguments; agents
are passed individually per call because the dispatch predicates inspect
each agent's fields (e.g. ``critic._no_critiques_filed``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .console import console
from .llm import AgentResult, LLMResponse, ParseFailureError
from .providers import ContextTooLongError, is_transient
from .task import Task, TaskType
from .utils.categories import CompensationCategory as CC
from .validation import Violation, ViolationSeverity
from .workspace import log_scaffold_event

if TYPE_CHECKING:
    from .agents.critic import CriticAgent
    from .agents.computer import ComputerAgent
    from .agents.formatter import FormatterAgent
    from .agents.researcher import ResearcherAgent
    from .agents.reviewer import ReviewerAgent
    from .loop_state import LoopState
    from .metrics import MetricsTracker
    from .research_state import ResearchState
    from .workspace import WorkspaceManager


# Errors that indicate malformed LLM output (wrong types in tool args, etc.)
# rather than a true system failure. These are recoverable — skip the
# iteration rather than crashing the entire run.
_LLM_DATA_ERRORS = (TypeError, ValueError, KeyError, AttributeError)


def is_recoverable(exc: Exception) -> bool:
    """Return True if *exc* is safe to skip (transient API or malformed LLM data)."""
    if is_transient(exc):
        return True
    if isinstance(exc, _LLM_DATA_ERRORS):
        return True
    return False


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch(
    task: Task,
    iteration: int,
    research_state: ResearchState,
    loop_state: LoopState,
    *,
    researcher: ResearcherAgent,
    computer: ComputerAgent,
    reviewer: ReviewerAgent,
    critic: CriticAgent,
    formatter: FormatterAgent,
    on_compute_round,
    on_agent_round,
    print_call_summary,
) -> tuple[str, LLMResponse | AgentResult]:
    """Route *task* to the correct agent and return ``(agent_name, result)``."""
    tt = task.task_type

    if tt == TaskType.RESEARCH:
        console.print("[green]Researcher[/green] reasoning...")
        researcher.research_state = research_state
        result = researcher.run(task, iteration)
        loop_state.last_content_iteration = iteration
        return "researcher", result

    elif tt == TaskType.COMPUTE:
        console.print("[magenta]Computer[/magenta] computing...")
        computer.research_state = research_state
        result = computer.run(task, iteration, on_round=on_compute_round)
        loop_state.last_content_iteration = iteration
        return "computer", result

    elif tt == TaskType.REVIEW:
        console.print("[magenta]Reviewer[/magenta] reviewing...")
        reviewer.research_state = research_state
        result = reviewer.run(task, iteration, on_round=on_agent_round)
        loop_state.last_content_iteration = iteration
        return "reviewer", result

    elif tt == TaskType.FORMAT:
        console.print("[cyan]Formatter[/cyan] producing ANSWER.md...")
        formatter.research_state = research_state
        result = formatter.run(task, iteration)
        print_call_summary(result)
        return "formatter", result

    elif tt == TaskType.CRITIQUE:
        console.print("[red]Deep Critic[/red] reviewing...")
        critic.research_state = research_state
        response = critic.run(task, iteration, on_round=on_agent_round)
        if critic._no_critiques_filed:
            console.print("[dim]Critic: no issues found[/dim]")
        else:
            crits = list(research_state.critiques.values())
            recent = [c for c in crits if c.iteration_filed == iteration]
            if recent:
                console.print(f"[red]Critic filed {len(recent)} critique(s)[/red]")
        return "deep_critic", response

    else:
        console.print(
            f"[yellow]Unknown task type '{tt}', defaulting to researcher[/yellow]"
        )
        researcher.research_state = research_state
        result = researcher.run(task, iteration)
        return "researcher", result


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


def handle_context_too_long(
    task: Task,
    exc: ContextTooLongError,
    iteration: int,
    loop_state: LoopState,
    workspace: WorkspaceManager,
    metrics: MetricsTracker,
) -> None:
    """Log context-too-long error and record for orchestrator — do not crash."""
    metrics.alert(
        iteration,
        f"Context too long for {task.task_id}: {exc.input_tokens} input tokens "
        f"(model limit {exc.max_context})",
    )
    loop_state.agent_failures.append(
        {
            "task_id": task.task_id,
            "agent": task.task_type.value,
            "event": "context_too_long",
            "detail": (
                f"Context exceeded model limit (~{exc.input_tokens} input tokens, "
                f"limit {exc.max_context}). Simplify or decompose the task."
            ),
            "iteration": iteration,
        }
    )
    console.print(
        f"[yellow]Context too long for {task.task_id} — skipping "
        f"(~{exc.input_tokens} input tokens, limit {exc.max_context})[/yellow]"
    )
    log_scaffold_event(
        workspace.root,
        iteration,
        CC.LOOP_CONTROL,
        "context_too_long",
        f"task={task.task_id}, input_tokens={exc.input_tokens}, "
        f"max_context={exc.max_context}",
    )


def handle_dispatch_error(
    task: Task,
    exc: Exception,
    iteration: int,
    loop_state: LoopState,
    workspace: WorkspaceManager,
    metrics: MetricsTracker,
) -> None:
    """Handle transient dispatch errors — log and skip to next iteration."""
    metrics.alert(
        iteration,
        f"Dispatch failed (transient): {type(exc).__name__}: {exc}",
    )
    loop_state.pending_violations.append(
        Violation(
            check="dispatch_failure",
            severity=ViolationSeverity.WARNING,
            message=(
                f"Agent dispatch failed with transient error: "
                f"{type(exc).__name__}: {str(exc)[:200]}"
            ),
        )
    )
    console.print(f"[yellow]Dispatch failed (transient), skipping: {exc}[/yellow]")
    log_scaffold_event(
        workspace.root,
        iteration,
        CC.LOOP_CONTROL,
        "dispatch_failure",
        f"{type(exc).__name__}: {str(exc)[:200]}",
    )


def handle_parse_failure(
    task: Task,
    exc: ParseFailureError,
    iteration: int,
    loop_state: LoopState,
    workspace: WorkspaceManager,
    metrics: MetricsTracker,
) -> None:
    """Handle evidence agent parse failure — log and report to orchestrator.

    No evidence is stored. The failure is appended to ``agent_failures``
    so the orchestrator can decide whether to re-dispatch or decompose.
    """
    metrics.alert(
        iteration,
        f"Parse failure on {exc.agent_name}: {exc.detail}",
    )
    loop_state.agent_failures.append(
        {
            "task_id": task.task_id,
            "agent": exc.agent_name,
            "event": "parse_failure",
            "detail": (
                f"Agent {exc.agent_name} failed to produce valid structured "
                f"output after retries. No evidence was stored. "
                f"Consider re-dispatching or decomposing the task."
            ),
            "iteration": iteration,
        }
    )
    console.print(
        f"[yellow]{exc.agent_name}: parse failure — no evidence stored. "
        f"Reporting to orchestrator.[/yellow]"
    )
    log_scaffold_event(
        workspace.root,
        iteration,
        CC.OUTPUT_NORMALIZATION,
        "parse_failure_no_evidence",
        f"agent={exc.agent_name}, task={task.task_id}, {exc.detail[:200]}",
    )


def record_agent_failures(
    task: Task,
    agent_name: str,
    result,
    iteration: int,
    loop_state: LoopState,
    workspace: WorkspaceManager,
) -> None:
    """Inspect agent result for failure signals and record for orchestrator context."""
    stop = getattr(result, "stop_reason", None)

    # max_tokens truncation (one-shot or agentic)
    if stop == "max_tokens":
        out_tok = (
            getattr(result, "output_tokens", None)
            or getattr(result, "total_output_tokens", None)
            or 0
        )
        loop_state.agent_failures.append(
            {
                "task_id": task.task_id,
                "agent": agent_name,
                "event": "max_tokens_truncation",
                "detail": (
                    f"Output hit token limit ({out_tok} tokens). "
                    f"Decompose into smaller subtasks, each targeting a single "
                    f"derivation step or sub-claim."
                ),
                "iteration": iteration,
            }
        )
        log_scaffold_event(
            workspace.root,
            iteration,
            CC.LOOP_CONTROL,
            "agent_failure_max_tokens",
            f"task={task.task_id}, agent={agent_name}",
        )

    # max_rounds exhaustion (agentic only)
    if stop == "max_rounds_forced" and isinstance(result, AgentResult):
        loop_state.agent_failures.append(
            {
                "task_id": task.task_id,
                "agent": agent_name,
                "event": "max_rounds_exhaustion",
                "detail": f"Exhausted {result.rounds} tool-use rounds without completing.",
                "iteration": iteration,
            }
        )
        log_scaffold_event(
            workspace.root,
            iteration,
            CC.LOOP_CONTROL,
            "agent_failure_max_rounds",
            f"task={task.task_id}, agent={agent_name}, rounds={result.rounds}",
        )
