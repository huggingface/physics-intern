#!/usr/bin/env python3
"""Run all 70 CritPt benchmark problems through the one-shot baseline with rolling parallelism.

Each problem = one LLM call via `open_dirac.one_shot`. Produces CritPt-format
submission JSONs progressively. Supports resume from interrupted runs.

Usage:
    uv run python scripts/run_critpt_oneshot.py
    uv run python scripts/run_critpt_oneshot.py --model claude-4.6-opus --concurrency 5
    uv run python scripts/run_critpt_oneshot.py --problems 1-10
    uv run python scripts/run_critpt_oneshot.py --resume results/critpt_oneshot/model/run_dir/
    uv run python scripts/run_critpt_oneshot.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_critpt_common import (
    PROJECT_ROOT,
    DEFAULT_PROBLEMS_DIR,
    DEFAULTS,
    Problem,
    RunResult,
    resolve_critpt_model_string,
    discover_problems,
    load_resume_config,
    find_completed_submissions,
    resolve_model,
    make_output_dir,
    write_submission_json,
    write_batch_metadata,
    write_initial_batch_metadata,
    save_raw_response,
    setup_signal_handler,
    print_final_summary,
)
from open_dirac.verification.evaluate import extract_answer_code  # noqa: E402

DEFAULT_RESULTS_BASE = PROJECT_ROOT / "results" / "critpt_oneshot"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run CritPt benchmark problems through one-shot baseline in parallel.",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an existing output directory (recovers all params)",
    )
    p.add_argument(
        "--model",
        default=None,
        help=f"Model key from models.yaml (default: {DEFAULTS['model']})",
    )
    p.add_argument(
        "--concurrency", type=int, default=10, help="Max parallel runs (default: 10)"
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-problem timeout in seconds (default: 1800)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for submission JSONs",
    )
    p.add_argument(
        "--problems-dir",
        type=Path,
        default=DEFAULT_PROBLEMS_DIR,
        help="Directory of problem YAMLs",
    )
    p.add_argument(
        "--problems",
        type=str,
        default=None,
        help='Subset of problems, e.g. "1-10" or "1,5,30-40"',
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run problems even if submission JSON already exists",
    )
    p.add_argument(
        "--no-sibling-history",
        action="store_true",
        help="Do not fold prior attempts from sibling output dirs "
        "into batch_metadata.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    return p


# ---------------------------------------------------------------------------
# Stderr stats parser (one-shot format)
# ---------------------------------------------------------------------------


def _parse_stderr_stats(stderr: str) -> dict | None:
    """Best-effort extraction of token/cost stats from one-shot stderr."""
    stats: dict = {}
    for line in stderr.splitlines():
        try:
            if "Input tokens:" in line:
                stats["input_tokens"] = int(line.split(":")[-1].strip())
            elif "Output tokens:" in line:
                stats["output_tokens"] = int(line.split(":")[-1].strip())
            elif "Reasoning:" in line and "tokens" not in line.split(":")[-1]:
                stats["reasoning_tokens"] = int(line.split(":")[-1].strip())
            elif "Est. cost:" in line:
                stats["cost_usd"] = float(line.split("$")[-1].strip())
        except (ValueError, IndexError):
            pass
    return stats if stats else None


# ---------------------------------------------------------------------------
# Worker: run one problem
# ---------------------------------------------------------------------------


async def run_one_problem(
    problem: Problem,
    model_key: str,
    timeout: float,
    semaphore: asyncio.Semaphore,
    logs_dir: Path | None,
) -> RunResult:
    """Run a single CritPt problem via one-shot subprocess."""
    async with semaphore:
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "open_dirac.one_shot",
            str(problem.yaml_path),
            "--model",
            model_key,
        ]

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
                start_new_session=True,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            elapsed = time.monotonic() - start

            stdout_text = stdout.decode(errors="replace")
            stderr_text = stderr.decode(errors="replace")

            # Extract answer code from LLM response
            if stdout_text.strip() and "```python" not in stdout_text:
                fenced = f"```python\n{stdout_text}\n```"
            else:
                fenced = stdout_text
            answer_code = extract_answer_code(fenced) if stdout_text.strip() else None

            stats = _parse_stderr_stats(stderr_text)

            success = answer_code is not None
            error = None
            if not success:
                if proc.returncode != 0:
                    error = f"exit code {proc.returncode}: {stderr_text[-500:]}"
                else:
                    error = "no answer code found in response"

            save_raw_response(logs_dir, problem, stdout_text, stderr_text, success)

            return RunResult(
                problem_n=problem.n,
                problem_id=problem.problem_id,
                success=success,
                answer_code=answer_code,
                error=error,
                duration_s=elapsed,
                stats=stats,
                returncode=proc.returncode,
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
            return RunResult(
                problem_n=problem.n,
                problem_id=problem.problem_id,
                success=False,
                answer_code=None,
                error=f"timeout after {timeout:.0f}s",
                duration_s=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return RunResult(
                problem_n=problem.n,
                problem_id=problem.problem_id,
                success=False,
                answer_code=None,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=elapsed,
            )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run_batch(args: argparse.Namespace) -> int:
    """Main batch orchestrator. Returns exit code."""
    # Handle --resume: restore params from saved batch_metadata.json
    if args.resume:
        if not args.resume.is_dir():
            print(f"Error: resume directory not found: {args.resume}", file=sys.stderr)
            return 1
        gen_cfg, run_cfg = load_resume_config(args.resume)
        args.output_dir = args.resume
        if args.model is None:
            args.model = gen_cfg.get("model_key")
        if run_cfg.get("problems_dir"):
            args.problems_dir = Path(run_cfg["problems_dir"])
        if run_cfg.get("problems_subset"):
            args.problems = run_cfg["problems_subset"]
        print(f"Resuming from {args.resume}", file=sys.stderr)

    resolve_model(args, args.output_dir)

    critpt_model = resolve_critpt_model_string(args.model)

    problems = discover_problems(args.problems_dir, args.problems)
    if not problems:
        print("Error: no problems found", file=sys.stderr)
        return 1

    output_dir = make_output_dir(args, DEFAULT_RESULTS_BASE, create=not args.dry_run)
    logs_dir = output_dir / "logs"
    if not args.dry_run:
        logs_dir.mkdir(exist_ok=True)

    # Resume: skip completed
    n_skip = 0
    if not args.force:
        completed = find_completed_submissions(output_dir)
        before = len(problems)
        problems = [p for p in problems if p.n not in completed]
        n_skip = before - len(problems)

    # Print plan
    print(f"Model:       {args.model} ({critpt_model})", file=sys.stderr)
    print(
        f"Problems:    {len(problems) + n_skip} total, "
        f"{n_skip} skipped, {len(problems)} to run",
        file=sys.stderr,
    )
    print(f"Concurrency: {args.concurrency}", file=sys.stderr)
    print(f"Timeout:     {args.timeout}s per problem", file=sys.stderr)
    print(f"Output:      {output_dir}", file=sys.stderr)
    print("---", file=sys.stderr)

    if not problems:
        print("All problems already completed.", file=sys.stderr)
        return 0

    if args.dry_run:
        for p in problems:
            print(f"  {p.problem_id}", file=sys.stderr)
        return 0

    generation_config = {
        "system": "open_dirac_one_shot",
        "model_key": args.model,
        "use_python": False,
        "use_web_search": False,
        "use_golden_for_prev_steps": False,
        "parsing": False,
        "multiturn_with_answer": False,
    }
    run_config = {
        "problems_dir": str(args.problems_dir),
        "problems_subset": args.problems,
    }

    # Run with semaphore-controlled concurrency
    semaphore = asyncio.Semaphore(args.concurrency)
    start_time = datetime.now(timezone.utc)
    # Stub metadata so a killed run is resumable before any workers finish.
    write_initial_batch_metadata(
        output_dir,
        critpt_model,
        generation_config,
        run_config,
        start_time,
    )
    total = len(problems)
    completed_count = 0
    succeeded = 0
    failed = 0
    all_results: list[RunResult] = []
    lock = asyncio.Lock()

    async def worker(problem: Problem) -> RunResult:
        nonlocal completed_count, succeeded, failed

        result = await run_one_problem(
            problem,
            args.model,
            args.timeout,
            semaphore,
            logs_dir,
        )

        if result.success:
            write_submission_json(result, output_dir, critpt_model, generation_config)

        async with lock:
            completed_count += 1
            if result.success:
                succeeded += 1
            else:
                failed += 1
            all_results.append(result)

            status = "OK" if result.success else f"FAIL: {result.error}"
            cost_str = ""
            if result.stats and result.stats.get("cost_usd"):
                cost_str = f", ${result.stats['cost_usd']:.4f}"
            print(
                f"[{completed_count}/{total}] C{result.problem_n} "
                f"({result.duration_s:.0f}s{cost_str}) {status}",
                file=sys.stderr,
            )

        return result

    tasks = [asyncio.create_task(worker(p)) for p in problems]

    loop = asyncio.get_running_loop()
    setup_signal_handler(loop, tasks)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(results):
        if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
            p = problems[i]
            async with lock:
                all_results.append(
                    RunResult(
                        problem_n=p.n,
                        problem_id=p.problem_id,
                        success=False,
                        answer_code=None,
                        error=str(r),
                        duration_s=0,
                    )
                )
                failed += 1
                completed_count += 1

    end_time = datetime.now(timezone.utc)

    write_batch_metadata(
        output_dir,
        critpt_model,
        all_results,
        generation_config,
        run_config,
        start_time,
        end_time,
        include_sibling_history=not args.resume and not args.no_sibling_history,
    )
    print_final_summary(
        all_results,
        total,
        succeeded,
        failed,
        start_time,
        end_time,
        output_dir,
    )

    return 0 if failed == 0 else 1


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(run_batch(args)))


if __name__ == "__main__":
    main()
