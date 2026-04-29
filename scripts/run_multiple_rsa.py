#!/usr/bin/env python3
"""Run open_dirac_rsa N times on a single problem with concurrency control.

Reports pass@k results: how many RSA runs produced a correct majority-vote
answer, based on VERIFICATION.md formal evaluation in each workspace (same
pattern as run_multiple.py and run_multiple_autophysicist.py). RSA-specific
metrics (rounds, majority vote, per-round costs) are enriched from the
workspace's rsa_result.json.

Usage:
    uv run python scripts/run_multiple_rsa.py problems/critpt/yaml/Challenge_1_main.yaml --runs 10
    uv run python scripts/run_multiple_rsa.py problems/critpt/yaml/Challenge_1_main.yaml --runs 10 --concurrency 3
    uv run python scripts/run_multiple_rsa.py problems/critpt/yaml/Challenge_1_main.yaml --runs 20 --concurrency 3 --model claude-4.6-opus -N 6 -K 2 -T 4
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
    # RSA-specific (parsed from rsa_result.json)
    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    total_calls: int = 0
    majority_vote: dict | None = None
    rounds: list | None = None


# ---------------------------------------------------------------------------
# Workspace result parsing
# ---------------------------------------------------------------------------


def parse_workspace_results(workspace_dir: Path) -> tuple[str, bool, dict | None]:
    """Read formal verification results from a completed workspace.

    Returns (formal_answer, has_answer, rsa_payload).
    """
    # Check ANSWER.md
    answer_path = workspace_dir / "ANSWER.md"
    has_answer = False
    if answer_path.exists():
        raw = answer_path.read_text().strip()
        if raw and not raw.startswith("FORMATTER_REJECTION"):
            has_answer = True

    # Check VERIFICATION.md (frontmatter format, same as vanilla/autophysicist)
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

    # Parse rsa_result.json for RSA-specific metrics
    rsa_payload = None
    rsa_json = workspace_dir / "rsa_result.json"
    if rsa_json.exists():
        try:
            rsa_payload = json.loads(rsa_json.read_text())
        except Exception:
            pass

    return formal_answer, has_answer, rsa_payload


# ---------------------------------------------------------------------------
# Worker: run one RSA instance
# ---------------------------------------------------------------------------

_ROUND_RE = re.compile(r"Round\s+(\d+)/(\d+)")


async def run_one(
    run_index: int,
    problem_path: Path,
    model_key: str | None,
    config_path: Path | None,
    rsa_N: int | None,
    rsa_K: int | None,
    rsa_T: int | None,
    rsa_concurrency: int | None,
    workspace_dir: Path,
    timeout: float,
    semaphore: asyncio.Semaphore,
    print_lock: asyncio.Lock,
) -> RunResult:
    """Run a single open_dirac_rsa subprocess, streaming round progress."""
    cmd = [
        "uv",
        "run",
        "open_dirac_rsa",
        str(problem_path),
        "--workspace-dir",
        str(workspace_dir),
    ]
    if model_key:
        cmd.extend(["--model", model_key])
    if config_path:
        cmd.extend(["--config", str(config_path)])
    if rsa_N is not None:
        cmd.extend(["-N", str(rsa_N)])
    if rsa_K is not None:
        cmd.extend(["-K", str(rsa_K)])
    if rsa_T is not None:
        cmd.extend(["-T", str(rsa_T)])
    if rsa_concurrency is not None:
        cmd.extend(["--concurrency", str(rsa_concurrency)])

    async with semaphore:
        start = time.monotonic()
        try:
            # PYTHONUNBUFFERED: flush stderr per-line so we can detect
            # "Round X/Y" progress markers via readline().
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
                start_new_session=True,
                env=env,
            )

            stderr_tail: list[str] = []

            async def _drain_stdout():
                """RSA writes the winning response to stdout — discard it
                (the workspace already has ANSWER.md), just keep the pipe
                drained to avoid the child blocking on a full buffer."""
                assert proc.stdout is not None
                await proc.stdout.read()

            async def _stream_stderr():
                """Detect 'Round X/Y' markers and print progress; retain the
                tail for error reporting."""
                assert proc.stderr is not None
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break
                    text = line.decode(errors="replace")
                    stderr_tail.append(text)
                    if len(stderr_tail) > 50:
                        del stderr_tail[:-50]
                    m = _ROUND_RE.search(text)
                    if m:
                        elapsed_so_far = time.monotonic() - start
                        async with print_lock:
                            print(
                                f"  run{run_index:03d}  round {m.group(1)}/{m.group(2)}  "
                                f"({elapsed_so_far:.0f}s)",
                                file=sys.stderr,
                            )

            await asyncio.wait_for(
                asyncio.gather(_drain_stdout(), _stream_stderr(), proc.wait()),
                timeout=timeout,
            )
            elapsed = time.monotonic() - start

            formal_answer, has_answer, rsa_payload = parse_workspace_results(
                workspace_dir
            )

            error = None
            if proc.returncode != 0 and not has_answer:
                tail = "".join(stderr_tail)[-500:]
                error = f"exit code {proc.returncode}: {tail}"

            return _build_result(
                run_index,
                workspace_dir,
                formal_answer,
                has_answer,
                elapsed,
                proc.returncode,
                error,
                rsa_payload,
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
            # Salvage whatever the child managed to write before the timeout.
            formal_answer, has_answer, rsa_payload = parse_workspace_results(
                workspace_dir
            )
            if not has_answer:
                formal_answer = "error"
            return _build_result(
                run_index,
                workspace_dir,
                formal_answer,
                has_answer,
                elapsed,
                None,
                f"timeout after {timeout:.0f}s" if not has_answer else None,
                rsa_payload,
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


def _build_result(
    run_index: int,
    workspace_dir: Path,
    formal_answer: str,
    has_answer: bool,
    elapsed: float,
    returncode: int | None,
    error: str | None,
    rsa_payload: dict | None,
) -> RunResult:
    """Build a RunResult, pulling RSA-specific metrics from the workspace JSON."""
    tokens = (rsa_payload or {}).get("tokens") or {}
    rsa_params = (rsa_payload or {}).get("rsa_params") or {}
    N = rsa_params.get("N") or 0
    T = rsa_params.get("T") or 0
    return RunResult(
        run_index=run_index,
        workspace_dir=workspace_dir,
        formal_answer=formal_answer,
        has_answer=has_answer,
        duration_s=round(elapsed, 1),
        returncode=returncode,
        error=error,
        total_cost_usd=(rsa_payload or {}).get("total_cost_usd", 0.0),
        input_tokens=tokens.get("input", 0),
        output_tokens=tokens.get("output", 0),
        reasoning_tokens=tokens.get("reasoning", 0),
        answer_tokens=tokens.get("answer", 0),
        total_calls=N * T,
        majority_vote=(rsa_payload or {}).get("majority_vote"),
        rounds=(rsa_payload or {}).get("rounds"),
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run_multiple(args: argparse.Namespace) -> int:
    """Run N RSA instances and report aggregate results."""
    n = args.runs
    concurrency = args.concurrency
    semaphore = asyncio.Semaphore(concurrency)

    # Generate workspace directories with a shared timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = (args.model or DEFAULTS["model"]).replace("/", "-").replace(":", "-")
    problem_stem = args.problem.stem
    workspace_base = args.workspace_base

    workspace_dirs = [
        workspace_base / f"{timestamp}_{problem_stem}_{safe_model}_rsa_run{i:03d}"
        for i in range(n)
    ]

    print(f"Problem:     {args.problem.name}", file=sys.stderr)
    if args.model:
        print(f"Model:       {args.model}", file=sys.stderr)
    rsa_parts: list[str] = []
    if args.N is not None:
        rsa_parts.append(f"N={args.N}")
    if args.K is not None:
        rsa_parts.append(f"K={args.K}")
    if args.T is not None:
        rsa_parts.append(f"T={args.T}")
    if rsa_parts:
        print(f"RSA params:  {', '.join(rsa_parts)}", file=sys.stderr)
    print(f"Runs:        {n}", file=sys.stderr)
    print(f"Concurrency: {concurrency} runs", file=sys.stderr)
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
            args.config,
            args.N,
            args.K,
            args.T,
            args.rsa_concurrency,
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
            cost_str = (
                f", ${result.total_cost_usd:.4f}" if result.total_cost_usd else ""
            )
            print(
                f"[{completed}/{n}] run{run_index:03d} ({result.duration_s:.0f}s{cost_str}) {status}",
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

    all_results.sort(key=lambda r: r.run_index)

    # Count outcomes
    counts: dict[str, int] = {}
    for r in all_results:
        counts[r.formal_answer] = counts.get(r.formal_answer, 0) + 1

    total_cost = sum(r.total_cost_usd for r in all_results)
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
    if total_cost > 0:
        print(f"Total cost: ${total_cost:.4f}", file=sys.stderr)
    print(f"Wall clock: {wall_clock:.0f}s", file=sys.stderr)

    # Write summary JSON
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{timestamp}_{problem_stem}_{safe_model}_rsa.json"
    output_path = output_dir / filename

    payload = {
        "mode": "rsa",
        "problem": problem_stem,
        "problem_path": str(args.problem),
        "model": args.model or DEFAULTS["model"],
        "rsa_params": {"N": args.N, "K": args.K, "T": args.T},
        "num_runs": n,
        "concurrency": concurrency,
        "timeout_s": args.timeout,
        "timestamp": timestamp,
        "summary": counts,
        "total_cost_usd": round(total_cost, 6),
        "wall_clock_s": round(wall_clock, 1),
        "runs": [
            {
                "run_index": r.run_index,
                "workspace": str(r.workspace_dir),
                "formal_answer": r.formal_answer,
                "has_answer": r.has_answer,
                "duration_s": r.duration_s,
                "total_cost_usd": r.total_cost_usd,
                "total_calls": r.total_calls,
                "tokens": {
                    "input": r.input_tokens,
                    "output": r.output_tokens,
                    "reasoning": r.reasoning_tokens,
                    "answer": r.answer_tokens,
                },
                "majority_vote": r.majority_vote,
                "rounds": r.rounds,
                "returncode": r.returncode,
                "error": r.error,
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
        description="Run open_dirac_rsa N times on a single problem with concurrency.",
    )
    parser.add_argument("problem", type=Path, help="Path to problem YAML file")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model key from models.yaml (default: {DEFAULTS['model']})",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Config YAML file to pass through"
    )
    parser.add_argument(
        "-N",
        type=int,
        default=None,
        help="RSA population size (default: RSA runner default — 6)",
    )
    parser.add_argument(
        "-K",
        type=int,
        default=None,
        help="RSA aggregation subset size (default: RSA runner default — 2)",
    )
    parser.add_argument(
        "-T",
        type=int,
        default=None,
        help="RSA number of rounds (default: RSA runner default — 4)",
    )
    parser.add_argument(
        "--rsa-concurrency",
        type=int,
        default=None,
        help="Max parallel LLM calls within a single RSA run "
        "(default: RSA runner default — N)",
    )
    parser.add_argument(
        "--runs", type=int, required=True, help="Number of independent RSA runs"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max parallel RSA runs (default: 3). "
        "Note: each RSA run itself fans out up to N calls, "
        "so effective in-flight LLM calls ≈ concurrency × N.",
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
        default=PROJECT_ROOT / "results" / "multiple_rsa",
        help="Output directory for summary JSON",
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
