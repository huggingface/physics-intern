"""Console-side helpers: progress callbacks, task summaries, final report.

Pure printing helpers that have no dependency on engine state. Used both
directly from the engine loop and as callbacks passed into agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text

from .console import console
from .llm import AgentResult, LLMResponse

if TYPE_CHECKING:
    from .config import Config
    from .metrics import MetricsTracker
    from .task import Task


def fmt_duration(seconds: float) -> str:
    """Format a duration as e.g. '6.3s' or '2m05s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def print_task(task: Task) -> None:
    """Print a one-line task summary to the console."""
    text = Text()
    text.append("Task: ", style="bold")
    text.append(f"{task.task_id} ", style="cyan")
    text.append(f"[{task.task_type}] ", style="yellow")
    if task.target_claim:
        text.append(f"{task.target_claim} ", style="bold magenta")
    text.append(f"-> {task.assigned_to}", style="green")
    console.print(text)


def on_round_progress(
    round_num,
    stop_reason,
    tool_calls,
    total_input,
    total_output,
    round_input,
    round_output,
    round_duration,
    round_reasoning=0,
    round_answer=0,
) -> None:
    """Progress callback for agent tool-use rounds."""
    if round_reasoning:
        tokens = f"{round_input:,}in + {round_output:,}out ({round_reasoning:,}r + {round_answer:,}a)"
    else:
        tokens = f"{round_input:,}in + {round_output:,}out"
    tps = f"{round_output / round_duration:,.0f} t/s" if round_duration > 0 else ""
    dur = fmt_duration(round_duration)
    detail = f"{tokens}, {dur}"
    if tps:
        detail += f", {tps}"
    if stop_reason == "forced_partial":
        console.print(
            f"  round {round_num}: forced final call ({detail})", style="dim magenta"
        )
        return
    n_tools = len(tool_calls)
    errors = sum(1 for tc in tool_calls if tc.is_error)
    if errors:
        status = f"{n_tools} tool call{'s' if n_tools != 1 else ''}, {errors} error{'s' if errors != 1 else ''}"
    else:
        status = f"{n_tools} tool call{'s' if n_tools != 1 else ''}"
    console.print(f"  round {round_num}: {status} ({detail})", style="dim magenta")


def print_call_summary(result) -> None:
    """Print a one-line timing/token summary for one-shot LLM calls."""
    if isinstance(result, AgentResult):
        out = result.total_output_tokens
        reasoning = result.total_reasoning_tokens
        answer = result.total_answer_tokens
        tokens = f"{result.total_input_tokens:,}in + {out:,}out"
        if reasoning:
            tokens += f" ({reasoning:,}r + {answer:,}a)"
    elif isinstance(result, LLMResponse):
        out = result.output_tokens
        reasoning = result.reasoning_tokens
        answer = result.answer_tokens
        tokens = f"{result.input_tokens:,}in + {out:,}out"
        if reasoning:
            tokens += f" ({reasoning:,}r + {answer:,}a)"
    else:
        return
    dur = fmt_duration(result.duration)
    tps = f", {out / result.duration:,.0f} t/s" if result.duration > 0 else ""
    console.print(f"  ({tokens}, {dur}{tps})", style="dim")


def print_final_report(
    iteration: int,
    metrics: MetricsTracker,
    config: Config,
    workspace_root,
) -> None:
    """Print the end-of-run summary (iterations, tokens, cost, alerts)."""
    console.rule("[bold green]SESSION COMPLETE[/bold green]")
    console.print(f"Total iterations: {iteration}")
    console.print(f"Total LLM calls: {len(metrics.calls)}")
    console.print(f"Total input tokens: {metrics.total_input_tokens:,}")
    console.print(f"Total output tokens: {metrics.total_output_tokens:,}")
    if config.input_cost or config.output_cost:
        cost = (
            metrics.total_input_tokens * config.input_cost
            + metrics.total_output_tokens * config.output_cost
        ) / 1_000_000
        console.print(f"Estimated cost: ${cost:.2f}")
    console.print(f"Workspace: {workspace_root.resolve()}")
    if metrics.alerts:
        console.print(f"\n[yellow]Alerts ({len(metrics.alerts)}):[/yellow]")
        for a in metrics.alerts[-5:]:
            console.print(f"  [iter {a['iteration']}] {a['message']}")
