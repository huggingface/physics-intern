"""Autophysicist: iterative Research Manager for theoretical physics.

Single-agent, stateless-iteration scaffolding. Each outer-loop iteration
rebuilds a fresh user message from PermanentMemory + the Scratchpad window
via :func:`_build_user_content`; no conversation history carries over
between iterations. The Manager's only durable state between iterations is
what it explicitly wrote to ``PERMANENT_MEMORY.md`` or ``SCRATCHPAD.md``.

The iteration counter is scaffolding-owned (persisted in ``.iteration`` and
bumped by :func:`_write_iteration_counter`), not LLM-decided — this lets
``--resume`` pick up exactly where a run left off. ``submit_final_answer``
sets ``problem_solved`` on the executor and breaks the outer loop;
:func:`_run_formal_verification` then runs once at the end of the run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import yaml  # noqa: E402

from ..baselines.cli import load_problem  # noqa: E402
from ..config import Config, DEFAULTS, build_config  # noqa: E402
from ..console import console  # noqa: E402
from ..llm import run_agent_loop, AgentResult  # noqa: E402
from ..metrics import MetricsTracker  # noqa: E402
from ..verification import (  # noqa: E402
    run_formal_evaluation,
    render_formal_evaluation,
    write_formal_eval_report,
)
from ..workspace import log_scaffold_event  # noqa: E402

from .memory import PermanentMemory, Scratchpad  # noqa: E402
from .tools import ManagerToolExecutor  # noqa: E402


def _load_system_prompt() -> str:
    """Load the Manager system prompt from the co-located prompt.md."""
    path = Path(__file__).parent / "prompt.md"
    return path.read_text()


def _build_user_content(
    problem_text: str,
    answer_template: str,
    permanent_memory: PermanentMemory,
    scratchpad: Scratchpad,
    iteration: int,
    max_iterations: int,
) -> str:
    """Assemble the user message for one Manager iteration."""
    mem_text = permanent_memory.read_full().strip()
    scratch_text = scratchpad.read_window().strip()

    parts = [f"# Iteration {iteration} of {max_iterations}"]

    parts.append(
        f"\n\n<problem_statement>\n{problem_text.strip()}\n</problem_statement>"
    )

    if answer_template.strip():
        parts.append(
            "\n\n<answer_template>\n"
            "**Answer template** — populate your final answer into this "
            "code template:\n\n"
            f"```python\n{answer_template.strip()}\n```\n"
            "</answer_template>"
        )

    if mem_text and mem_text != "# Permanent Memory":
        parts.append(f"\n\n<permanent_memory>\n{mem_text}\n</permanent_memory>")
    else:
        parts.append(
            "\n\n<permanent_memory>\n"
            "(Empty — no results recorded yet.)\n"
            "</permanent_memory>"
        )

    if scratch_text:
        parts.append(f"\n\n<scratchpad>\n{scratch_text}\n</scratchpad>")
    else:
        parts.append("\n\n<scratchpad>\n(Empty — no working notes yet.)\n</scratchpad>")

    return "".join(parts)


def _run_iteration(
    system_prompt: str,
    problem_text: str,
    answer_template: str,
    config: Config,
    permanent_memory: PermanentMemory,
    scratchpad: Scratchpad,
    workspace_root: Path,
    iteration: int,
    max_iterations: int,
    token_budget: int,
    tool_call_cap: int,
    max_rounds: int,
    sandbox_timeout: int,
    metrics: MetricsTracker,
) -> tuple[AgentResult, ManagerToolExecutor]:
    """Run one iteration of the Research Manager."""
    user_content = _build_user_content(
        problem_text,
        answer_template,
        permanent_memory,
        scratchpad,
        iteration,
        max_iterations,
    )

    executor = ManagerToolExecutor(
        config=config,
        permanent_memory=permanent_memory,
        scratchpad=scratchpad,
        workspace_root=workspace_root,
        iteration=iteration,
        token_budget=token_budget,
        tool_call_cap=tool_call_cap,
        sandbox_timeout=sandbox_timeout,
    )

    def on_round(
        round_num,
        stop_reason,
        round_tool_calls,
        total_input,
        total_output,
        round_input,
        round_output,
        **kwargs,
    ):
        executor.update_manager_tokens(total_input, total_output)

    result = run_agent_loop(
        system=system_prompt,
        user_content=user_content,
        config=config,
        tool_executor=executor,
        tools=ManagerToolExecutor.ALL_TOOLS,
        max_rounds=max_rounds,
        agent_name="manager",
        iteration=iteration,
        on_round=on_round,
    )

    metrics.record_call(
        iteration=iteration,
        agent="manager",
        input_tokens=result.total_input_tokens,
        output_tokens=result.total_output_tokens,
        duration=result.duration,
        max_tokens_hit=result.truncated,
        rounds=result.rounds,
        tool_calls=len(result.tool_calls),
        reasoning_tokens=result.total_reasoning_tokens,
        answer_tokens=result.total_answer_tokens,
    )

    return result, executor


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def _git_commit(workspace_root: Path, iteration: int, result: AgentResult) -> None:
    """Stage all and commit after each iteration."""
    if not (workspace_root / ".git").exists():
        return
    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(workspace_root),
        capture_output=True,
        check=False,
    )
    msg = (
        f"Iteration {iteration}: "
        f"{result.rounds} rounds, {len(result.tool_calls)} tool calls"
    )
    subprocess.run(
        ["git", "commit", "-m", msg, "--allow-empty"],
        cwd=str(workspace_root),
        capture_output=True,
        check=False,
    )


def _write_iteration_counter(workspace_root: Path, iteration: int) -> None:
    (workspace_root / ".iteration").write_text(str(iteration))


def _read_iteration_counter(workspace_root: Path) -> int:
    path = workspace_root / ".iteration"
    if path.exists():
        try:
            return int(path.read_text().strip())
        except ValueError:
            pass
    return 0


# TODO(slice-7): this block mirrors engine.py._run_formal_verification() and
# the simpler run/render/write sequence used by one_shot and rsa runners.
# Slice 7 should decide whether to extract a shared helper (into verification/
# or a new runner-utils module) rather than keep four near-duplicates.
def _run_formal_verification(workspace_root: Path, problem_path: Path) -> None:
    """Run formal (symbolic/numerical) answer evaluation at end of run.

    Mirrors engine.py._run_formal_verification().
    """
    problem_yaml = workspace_root / "problem.yaml"
    if not problem_yaml.exists():
        console.print(
            "[dim]Formal verification skipped: no problem.yaml in workspace[/]"
        )
        return

    try:
        with open(problem_yaml) as f:
            problem_def = yaml.safe_load(f)
    except Exception as exc:
        console.print(
            f"[yellow]Formal verification skipped: could not read problem.yaml: {exc}[/]"
        )
        return

    # Build reference lookup path from problem name
    problem_name = problem_def.get("name") if problem_def else None
    ref_lookup_path = Path(problem_name + ".yaml") if problem_name else None

    console.print("\n[bold]Formal answer evaluation...[/]")

    try:
        result = run_formal_evaluation(
            str(workspace_root),
            problem_def,
            problem_path=ref_lookup_path,
        )
        render_formal_evaluation(result)
        write_formal_eval_report(result, str(workspace_root))
        # Git commit the evaluation results
        if (workspace_root / ".git").exists():
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(workspace_root),
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["git", "commit", "-m", "Formal answer evaluation"],
                cwd=str(workspace_root),
                capture_output=True,
                check=False,
            )
    except Exception as exc:
        console.print(
            f"[yellow]Formal verification failed: {type(exc).__name__}: {exc}[/]"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="open_dirac.autophysicist",
        description="Autophysicist: iterative Research Manager for theoretical physics.",
    )
    parser.add_argument("problem", type=Path, help="Path to problem YAML file")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model key from models.yaml (default: {DEFAULTS['model']})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of Manager iterations (default: 50)",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=64_000,
        help="Token budget per iteration for wind-down trigger (default: 64000)",
    )
    parser.add_argument(
        "--tool-call-cap",
        type=int,
        default=15,
        help="Max tool calls per iteration (default: 15)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=30,
        help="Max LLM rounds per iteration (default: 30)",
    )
    parser.add_argument(
        "--scratchpad-window",
        type=int,
        default=5,
        help="Number of recent scratchpad entries to show (default: 5)",
    )
    parser.add_argument(
        "--sandbox-timeout",
        type=int,
        default=60,
        help="Code execution timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default=None,
        help="Workspace directory (default: auto-generated)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an existing workspace directory",
    )
    args = parser.parse_args()

    # --- Load problem ---
    problem_def, problem_text, answer_template = load_problem(args.problem)

    # --- Config ---
    config = build_config(args)

    # --- Workspace ---
    if args.resume:
        workspace_root = Path(args.resume)
        if not workspace_root.exists():
            print(f"Error: workspace not found: {workspace_root}", file=sys.stderr)
            sys.exit(1)
        start_iteration = _read_iteration_counter(workspace_root) + 1
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_problem = args.problem.stem.replace("/", "-")
        safe_model = config.model.replace("/", "-").replace(":", "-")
        workspace_root = Path(
            args.workspace_dir
            or f"workspaces/{timestamp}_{safe_problem}_{safe_model}_autophysicist"
        )
        workspace_root.mkdir(parents=True, exist_ok=True)
        start_iteration = 1

    config.workspace_dir = str(workspace_root)
    config.logs_dir = str(workspace_root / "logs")
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
    config.save(workspace_root)

    # --- Console log ---
    console.setup_log(workspace_root / "console.log")

    # --- Memory ---
    permanent_memory = PermanentMemory(workspace_root)
    scratchpad = Scratchpad(workspace_root, window_size=args.scratchpad_window)

    # --- System prompt ---
    system_prompt = _load_system_prompt()

    # --- Metrics ---
    metrics = MetricsTracker()

    # --- Git init (new workspace only) ---
    if not args.resume:
        subprocess.run(
            ["git", "init"],
            cwd=str(workspace_root),
            capture_output=True,
            check=False,
        )
        (workspace_root / "PROBLEM.md").write_text(f"# Problem\n\n{problem_text}\n")
        # Persist problem.yaml (with name field) for formal evaluation
        problem_data = dict(problem_def)
        problem_data["name"] = args.problem.stem
        with open(workspace_root / "problem.yaml", "w") as f:
            yaml.dump(problem_data, f, default_flow_style=False, sort_keys=False)
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(workspace_root),
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial workspace setup"],
            cwd=str(workspace_root),
            capture_output=True,
            check=False,
        )

    # --- Header ---
    console.print(f"[bold]Autophysicist[/bold] — {args.problem.name}")
    console.print(f"Model:      {config.model} ({config.model_id})")
    console.print(f"Provider:   {config.provider}")
    console.print(f"Workspace:  {workspace_root}")
    console.print(
        f"Iterations: up to {args.max_iterations}, "
        f"budget {args.token_budget:,} tokens/iter"
    )
    console.rule()

    # --- Outer iteration loop ---
    for iteration in range(start_iteration, args.max_iterations + 1):
        console.rule(f"[bold]Iteration {iteration}[/bold]")
        iter_start = time.time()

        try:
            result, executor = _run_iteration(
                system_prompt=system_prompt,
                problem_text=problem_text,
                answer_template=answer_template,
                config=config,
                permanent_memory=permanent_memory,
                scratchpad=scratchpad,
                workspace_root=workspace_root,
                iteration=iteration,
                max_iterations=args.max_iterations,
                token_budget=args.token_budget,
                tool_call_cap=args.tool_call_cap,
                max_rounds=args.max_rounds,
                sandbox_timeout=args.sandbox_timeout,
                metrics=metrics,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            break
        except Exception as exc:
            console.print(f"[red]Iteration {iteration} failed: {exc}[/red]")
            if config.workspace_dir:
                log_scaffold_event(
                    config.workspace_dir,
                    iteration,
                    "error",
                    "iteration_failed",
                    str(exc)[:500],
                )
            scratchpad.append(
                f"SYSTEM NOTE: Iteration {iteration} failed with error: {exc}",
                iteration,
            )
            _git_commit(workspace_root, iteration, AgentResult(text="(failed)"))
            _write_iteration_counter(workspace_root, iteration)
            continue

        iter_duration = time.time() - iter_start

        console.print(
            f"  Rounds: {result.rounds}, "
            f"Tool calls: {len(result.tool_calls)}, "
            f"Tokens: {result.total_input_tokens:,} in "
            f"/ {result.total_output_tokens:,} out, "
            f"Duration: {iter_duration:.1f}s, "
            f"Stop: {result.stop_reason}"
        )
        console.print(
            f"  Memory: {permanent_memory.size_chars:,} chars, "
            f"Scratchpad: {scratchpad.entry_count} entries"
        )

        _git_commit(workspace_root, iteration, result)
        _write_iteration_counter(workspace_root, iteration)
        (workspace_root / "METRICS.md").write_text(metrics.to_markdown())

        if executor.problem_solved:
            console.print(
                "[bold green]Problem solved![/bold green] Final answer submitted."
            )
            (workspace_root / "ANSWER.md").write_text(
                f"# Final Answer\n\n{executor.final_answer}\n"
            )
            break

    # --- Formal evaluation ---
    _run_formal_verification(workspace_root, args.problem)

    # --- Final summary ---
    console.rule("[bold]Run Complete[/bold]")
    total_iters = metrics.calls[-1].iteration if metrics.calls else 0
    console.print(f"Iterations completed: {total_iters}")
    console.print(
        f"Total tokens: {metrics.total_input_tokens:,} in "
        f"/ {metrics.total_output_tokens:,} out"
    )
    console.print(f"Workspace: {workspace_root}")
