#!/usr/bin/env python3
"""Run open_dirac_autophysicist N times on a single problem with concurrency control.

Reports pass@k results: how many runs produced a correct answer, based on
VERIFICATION.md formal evaluation in each workspace.

Usage:
    uv run python scripts/run_multiple_autophysicist.py problems/critpt/yaml/Challenge_1_main.yaml --runs 10
    uv run python scripts/run_multiple_autophysicist.py problems/critpt/yaml/Challenge_1_main.yaml --runs 10 --concurrency 3
    uv run python scripts/run_multiple_autophysicist.py problems/critpt/yaml/Challenge_1_main.yaml --runs 20 --concurrency 5 --model claude-4.6-opus --max-iterations 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_dirac.config import DEFAULTS  # noqa: E402
from open_dirac.utils.markdown import parse_frontmatter  # noqa: E402

DEFAULT_WORKSPACE_BASE = PROJECT_ROOT / "workspaces"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    run_index: int
    workspace_dir: Path
    formal_answer: str  # "correct", "incorrect", "inconclusive", "skipped", "no_verification", "no_answer", "error"
    has_answer: bool
    duration_s: float
    returncode: int | None = None
    error: str | None = None
    metrics: dict | None = None  # parsed from METRICS.md frontmatter


# ---------------------------------------------------------------------------
# Workspace result parsing
# ---------------------------------------------------------------------------


def parse_workspace_results(workspace_dir: Path) -> tuple[str, bool, dict | None]:
    """Read results from a completed workspace.

    Returns (formal_answer, has_answer, metrics_dict).
    """
    # Check ANSWER.md
    answer_path = workspace_dir / "ANSWER.md"
    has_answer = False
    if answer_path.exists():
        raw = answer_path.read_text().strip()
        if raw and not raw.startswith("FORMATTER_REJECTION"):
            has_answer = True

    # Check VERIFICATION.md
    formal_answer = "no_verification"
    verif_path = workspace_dir / "VERIFICATION.md"
    if verif_path.exists():
        try:
            fm, _ = parse_frontmatter(verif_path.read_text())
            fa = fm.get("formal_answer", "")
            if fa in ("correct", "incorrect", "inconclusive", "skipped"):
                formal_answer = fa
        except Exception:
            pass

    if not has_answer and formal_answer == "no_verification":
        formal_answer = "no_answer"

    # Parse METRICS.md
    metrics = None
    metrics_path = workspace_dir / "METRICS.md"
    if metrics_path.exists():
        try:
            fm, _ = parse_frontmatter(metrics_path.read_text())
            if fm:
                metrics = fm
        except Exception:
            pass

    return formal_answer, has_answer, metrics


# ---------------------------------------------------------------------------
# Worker: run one instance
# ---------------------------------------------------------------------------

_ITERATION_RE = re.compile(r"Iteration\s+(\d+)")


async def run_one(
    run_index: int,
    problem_path: Path,
    model_key: str | None,
    max_iterations: int | None,
    config_path: Path | None,
    token_budget: int | None,
    tool_call_cap: int | None,
    max_rounds: int | None,
    scratchpad_window: int | None,
    sandbox_timeout: int | None,
    workspace_dir: Path,
    timeout: float,
    semaphore: asyncio.Semaphore,
    print_lock: asyncio.Lock,
) -> RunResult:
    """Run a single open_dirac_autophysicist subprocess, streaming iteration progress."""
    cmd = [
        "uv",
        "run",
        "open_dirac_autophysicist",
        str(problem_path),
        "--workspace-dir",
        str(workspace_dir),
    ]
    if model_key:
        cmd.extend(["--model", model_key])
    if max_iterations is not None:
        cmd.extend(["--max-iterations", str(max_iterations)])
    if config_path:
        cmd.extend(["--config", str(config_path)])
    if token_budget is not None:
        cmd.extend(["--token-budget", str(token_budget)])
    if tool_call_cap is not None:
        cmd.extend(["--tool-call-cap", str(tool_call_cap)])
    if max_rounds is not None:
        cmd.extend(["--max-rounds", str(max_rounds)])
    if scratchpad_window is not None:
        cmd.extend(["--scratchpad-window", str(scratchpad_window)])
    if sandbox_timeout is not None:
        cmd.extend(["--sandbox-timeout", str(sandbox_timeout)])

    async with semaphore:
        start = time.monotonic()
        try:
            # PYTHONUNBUFFERED ensures stdout is flushed per-line so we
            # can detect iteration markers in real time via readline().
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
                start_new_session=True,
                env=env,
            )

            stderr_tail: list[str] = []  # keep last lines for error reporting

            async def _stream_stdout():
                """Read stdout line-by-line, detect iteration markers."""
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    text = line.decode(errors="replace")
                    m = _ITERATION_RE.search(text)
                    if m:
                        elapsed_so_far = time.monotonic() - start
                        async with print_lock:
                            print(
                                f"  run{run_index:03d}  iter {m.group(1)}  "
                                f"({elapsed_so_far:.0f}s)",
                                file=sys.stderr,
                            )

            async def _drain_stderr():
                """Collect stderr tail for error reporting."""
                assert proc.stderr is not None
                data = await proc.stderr.read()
                text = data.decode(errors="replace")
                for line in text.splitlines():
                    stderr_tail.append(line)
                if len(stderr_tail) > 50:
                    del stderr_tail[:-50]

            await asyncio.wait_for(
                asyncio.gather(_stream_stdout(), _drain_stderr(), proc.wait()),
                timeout=timeout,
            )
            elapsed = time.monotonic() - start

            formal_answer, has_answer, metrics = parse_workspace_results(workspace_dir)

            error = None
            if proc.returncode != 0 and not has_answer:
                tail = "".join(stderr_tail)[-500:]
                error = f"exit code {proc.returncode}: {tail}"

            return RunResult(
                run_index=run_index,
                workspace_dir=workspace_dir,
                formal_answer=formal_answer,
                has_answer=has_answer,
                duration_s=round(elapsed, 1),
                returncode=proc.returncode,
                error=error,
                metrics=metrics,
            )

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass
            # Still try to extract partial results
            formal_answer, has_answer, metrics = parse_workspace_results(workspace_dir)
            if not has_answer:
                formal_answer = "error"
            return RunResult(
                run_index=run_index,
                workspace_dir=workspace_dir,
                formal_answer=formal_answer,
                has_answer=has_answer,
                duration_s=round(elapsed, 1),
                error=f"timeout after {timeout:.0f}s" if not has_answer else None,
                metrics=metrics,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return RunResult(
                run_index=run_index,
                workspace_dir=workspace_dir,
                formal_answer="error",
                has_answer=False,
                duration_s=round(elapsed, 1),
                error=f"{type(exc).__name__}: {exc}",
            )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run_multiple(args: argparse.Namespace) -> int:
    """Run N autophysicist instances and report results."""
    n = args.runs
    concurrency = args.concurrency
    semaphore = asyncio.Semaphore(concurrency)

    # Generate workspace directories with shared timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = (args.model or DEFAULTS["model"]).replace("/", "-").replace(":", "-")
    problem_stem = args.problem.stem
    workspace_base = args.workspace_base

    workspace_dirs = []
    for i in range(n):
        ws_name = f"{timestamp}_{problem_stem}_{safe_model}_autophysicist_run{i:03d}"
        workspace_dirs.append(workspace_base / ws_name)

    print(f"Problem:     {args.problem.name}", file=sys.stderr)
    if args.model:
        print(f"Model:       {args.model}", file=sys.stderr)
    if args.max_iterations:
        print(f"Max iters:   {args.max_iterations}", file=sys.stderr)
    if args.token_budget:
        print(f"Token budget:{args.token_budget:,}", file=sys.stderr)
    print(f"Runs:        {n}", file=sys.stderr)
    print(f"Concurrency: {concurrency}", file=sys.stderr)
    print(f"Timeout:     {args.timeout}s per run", file=sys.stderr)
    print(f"Workspaces:  {workspace_base}/", file=sys.stderr)
    print("---", file=sys.stderr)

    start_time = datetime.now(timezone.utc)
    completed = 0
    all_results: list[RunResult] = []
    lock = asyncio.Lock()
    print_lock = asyncio.Lock()

    async def worker(run_index: int) -> RunResult:
        nonlocal completed
        result = await run_one(
            run_index,
            args.problem,
            args.model,
            args.max_iterations,
            args.config,
            args.token_budget,
            args.tool_call_cap,
            args.max_rounds,
            args.scratchpad_window,
            args.sandbox_timeout,
            workspace_dirs[run_index],
            args.timeout,
            semaphore,
            print_lock,
        )
        async with lock:
            completed += 1
            all_results.append(result)
            status = result.formal_answer.upper()
            if result.error:
                status = f"ERROR: {result.error[:80]}"
            print(
                f"[{completed}/{n}] run{run_index:03d} ({result.duration_s:.0f}s) {status}",
                file=sys.stderr,
            )
        return result

    tasks = [asyncio.create_task(worker(i)) for i in range(n)]

    # Ctrl+C handler
    loop = asyncio.get_running_loop()
    cancelled = False

    def _handler():
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            print("\nInterrupted — cancelling pending runs...", file=sys.stderr)
            for t in tasks:
                if not t.done():
                    t.cancel()

    loop.add_signal_handler(signal.SIGINT, _handler)

    await asyncio.gather(*tasks, return_exceptions=True)
    end_time = datetime.now(timezone.utc)

    # Sort by run_index
    all_results.sort(key=lambda r: r.run_index)

    # Count outcomes
    counts: dict[str, int] = {}
    for r in all_results:
        counts[r.formal_answer] = counts.get(r.formal_answer, 0) + 1

    wall_clock = (end_time - start_time).total_seconds()

    # Summary
    print("---", file=sys.stderr)
    parts = []
    if counts.get("correct"):
        parts.append(f"{counts['correct']}/{n} correct")
    if counts.get("incorrect"):
        parts.append(f"{counts['incorrect']} incorrect")
    if counts.get("inconclusive"):
        parts.append(f"{counts['inconclusive']} inconclusive")
    if counts.get("no_answer"):
        parts.append(f"{counts['no_answer']} no answer")
    if counts.get("no_verification"):
        parts.append(f"{counts['no_verification']} no verification")
    if counts.get("skipped"):
        parts.append(f"{counts['skipped']} skipped (no ground truth)")
    if counts.get("error"):
        parts.append(f"{counts['error']} errors")
    print(f"Results: {', '.join(parts) or '0 runs'}", file=sys.stderr)
    print(f"Wall clock: {wall_clock:.0f}s", file=sys.stderr)

    # Write JSON
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{timestamp}_{problem_stem}_{safe_model}_autophysicist.json"
    output_path = output_dir / filename

    payload = {
        "mode": "autophysicist",
        "problem": problem_stem,
        "problem_path": str(args.problem),
        "model": args.model or DEFAULTS["model"],
        "max_iterations": args.max_iterations,
        "token_budget": args.token_budget,
        "tool_call_cap": args.tool_call_cap,
        "max_rounds": args.max_rounds,
        "num_runs": n,
        "concurrency": concurrency,
        "timeout_s": args.timeout,
        "timestamp": timestamp,
        "summary": counts,
        "wall_clock_s": round(wall_clock, 1),
        "runs": [
            {
                "run_index": r.run_index,
                "workspace": str(r.workspace_dir),
                "formal_answer": r.formal_answer,
                "has_answer": r.has_answer,
                "duration_s": r.duration_s,
                "returncode": r.returncode,
                "error": r.error,
                "metrics": r.metrics,
            }
            for r in all_results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Saved to {output_path}", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run open_dirac_autophysicist N times on a single problem with concurrency.",
    )
    parser.add_argument("problem", type=Path, help="Path to problem YAML file")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model key from models.yaml (default: {DEFAULTS['model']})",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None, help="Max iterations per run"
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Config YAML file to pass through"
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Token budget per iteration (default: autophysicist default 64000)",
    )
    parser.add_argument(
        "--tool-call-cap",
        type=int,
        default=None,
        help="Max tool calls per iteration (default: autophysicist default 15)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Max LLM rounds per iteration (default: autophysicist default 30)",
    )
    parser.add_argument(
        "--scratchpad-window",
        type=int,
        default=None,
        help="Number of recent scratchpad entries to show (default: 5)",
    )
    parser.add_argument(
        "--sandbox-timeout",
        type=int,
        default=None,
        help="Code execution timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--runs", type=int, required=True, help="Number of independent runs"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Max parallel runs (default: 10)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-run timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--workspace-base",
        type=Path,
        default=DEFAULT_WORKSPACE_BASE,
        help="Base directory for workspaces",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "multiple_autophysicist",
        help="Output directory for results JSON",
    )
    args = parser.parse_args()

    if args.runs < 1:
        print("Error: --runs must be >= 1", file=sys.stderr)
        sys.exit(1)
    if not args.problem.exists():
        print(f"Error: problem file not found: {args.problem}", file=sys.stderr)
        sys.exit(1)

    sys.exit(asyncio.run(run_multiple(args)))


if __name__ == "__main__":
    main()
