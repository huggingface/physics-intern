#!/usr/bin/env python3
"""Run HLE-physics problems through the HLE one-shot baseline in parallel.

Each problem = one LLM call via ``physics_intern.hle_one_shot``. Produces an
HLE-native per-problem JSON progressively. Supports resume from interrupted
runs. Independent of the CritPt batch infrastructure.

Usage:
    uv run python scripts/run_hle_oneshot.py
    uv run python scripts/run_hle_oneshot.py --model claude-4.7-opus --concurrency 5
    uv run python scripts/run_hle_oneshot.py --problems 1-10
    uv run python scripts/run_hle_oneshot.py --resume results/hle_oneshot/<model>/<ts>/
    uv run python scripts/run_hle_oneshot.py --dry-run
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = PROJECT_ROOT / "src" / "physics_intern" / "models.yaml"
DEFAULT_PROBLEMS_DIR = PROJECT_ROOT / "problems" / "hle-physics"
DEFAULT_RESULTS_BASE = PROJECT_ROOT / "results" / "hle_oneshot"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from physics_intern.core.config import DEFAULTS  # noqa: E402


# ---------------------------------------------------------------------------
# Problem discovery
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    n: int  # 1-based index in alphabetical filename order
    problem_id: str  # the YAML's `id` field (24-char HLE hex)
    slug: str  # filename stem (e.g. ``66b727d3_antisymmetrized_...``)
    raw_subject: str
    yaml_path: Path


def parse_problem_range(range_str: str) -> set[int]:
    """Parse '1-10,15,30-40' into a set of ints."""
    result: set[int] = set()
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


def discover_problems(
    problems_dir: Path,
    problem_range: str | None = None,
) -> list[Problem]:
    """Return HLE problems numbered 1..N in alphabetical filename order."""
    problems: list[Problem] = []
    for i, p in enumerate(sorted(problems_dir.glob("*.yaml")), start=1):
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as exc:
            print(
                f"Warning: skipping {p.name}: YAML parse error: {exc}", file=sys.stderr
            )
            continue
        pid = data.get("id") or p.stem
        problems.append(
            Problem(
                n=i,
                problem_id=str(pid),
                slug=p.stem,
                raw_subject=str(data.get("raw_subject", "")),
                yaml_path=p.resolve(),
            )
        )
    if problem_range:
        wanted = parse_problem_range(problem_range)
        problems = [p for p in problems if p.n in wanted]
    return problems


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    problem_n: int
    problem_id: str
    slug: str
    raw_subject: str
    success: bool
    response_text: str | None
    answer_code: str | None
    formal_eval: dict | None  # {"verdict": str, "method": str|None, "error": str|None}
    error: str | None
    duration_s: float
    stats: dict = field(default_factory=dict)
    returncode: int | None = None


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def resolve_provider_model_string(model_key: str) -> str:
    """Convert PhysicsIntern model key to ``provider/model_id`` for metadata."""
    if not MODELS_YAML.exists():
        return model_key
    try:
        registry = yaml.safe_load(MODELS_YAML.read_text())
    except yaml.YAMLError:
        return model_key
    entry = (registry or {}).get(model_key)
    if entry:
        model_id = entry.get("model_id", model_key)
        return f"{entry['provider']}/{model_id}"
    return model_key


def _read_model_from_output_dir(output_dir: Path) -> str | None:
    """Try to recover the model key from a previous run's metadata."""
    meta_path = output_dir / "batch_metadata.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text())
            return data.get("generation_config", {}).get("model_key")
        except (json.JSONDecodeError, OSError):
            pass
    for f in output_dir.glob("*.json"):
        if f.name == "batch_metadata.json":
            continue
        try:
            data = json.loads(f.read_text())
            mk = data.get("generation_config", {}).get("model_key")
            if mk:
                return mk
        except (json.JSONDecodeError, OSError):
            continue
    return None


def resolve_model(args: argparse.Namespace, output_dir: Path | None) -> str:
    """Resolve model with precedence: --model > resumed > default."""
    if args.model is None and output_dir and output_dir.exists():
        recovered = _read_model_from_output_dir(output_dir)
        if recovered:
            args.model = recovered
            print(f"Resumed model from previous run: {recovered}", file=sys.stderr)
    if args.model is None:
        args.model = DEFAULTS["model"]
    return args.model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run HLE-physics problems through the HLE one-shot baseline.",
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
        help="Output directory for per-problem JSONs",
    )
    p.add_argument(
        "--problems-dir",
        type=Path,
        default=DEFAULT_PROBLEMS_DIR,
        help="Directory of HLE problem YAMLs",
    )
    p.add_argument(
        "--problems",
        type=str,
        default=None,
        help='Subset of problems, e.g. "1-10" or "1,5,30-40" (1-based index in '
        "sorted filename order)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run problems even if a per-problem JSON already exists",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    return p


# ---------------------------------------------------------------------------
# Stderr parsing (mirrors the one-shot runner's stderr format)
# ---------------------------------------------------------------------------


_EVAL_RE = re.compile(r"^Evaluation:\s+(\S+)(?:\s+\(([^)]*)\))?", re.MULTILINE)


def _parse_stderr_stats(stderr: str) -> dict:
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
    return stats


def _parse_eval_verdict(stderr: str) -> dict | None:
    """Pull the ``Evaluation: …`` line from the one-shot stderr.

    Returns ``{"verdict": "correct"|"incorrect"|"skipped"|"error",
    "method": str|None, "error": str|None}`` or None when no such line was
    emitted (e.g. the subprocess crashed before formal eval ran).
    """
    m = _EVAL_RE.search(stderr)
    if not m:
        return None
    verdict_raw = m.group(1).upper()
    detail = (m.group(2) or "").strip() or None
    verdict_map = {
        "CORRECT": "correct",
        "INCORRECT": "incorrect",
        "SKIPPED": "skipped",
        "ERROR": "error",
    }
    verdict = verdict_map.get(verdict_raw, verdict_raw.lower())
    out: dict = {"verdict": verdict}
    if verdict in ("correct", "incorrect"):
        out["method"] = detail
        out["error"] = None
    elif verdict == "skipped":
        out["method"] = None
        out["error"] = None
        out["skip_reason"] = detail
    else:
        out["method"] = None
        out["error"] = detail
    return out


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
    """Run a single HLE problem via the hle_one_shot subprocess."""
    async with semaphore:
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "physics_intern.hle_one_shot",
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
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            elapsed = time.monotonic() - start

            stdout_text = stdout.decode(errors="replace")
            stderr_text = stderr.decode(errors="replace")

            response_text = stdout_text if stdout_text.strip() else None

            # HLE responses are plain Explanation / Answer / Confidence text;
            # no python code block is extracted.
            answer_code = response_text

            stats = _parse_stderr_stats(stderr_text)
            formal_eval = _parse_eval_verdict(stderr_text)

            success = response_text is not None and proc.returncode == 0
            error: str | None = None
            if not success:
                if proc.returncode != 0:
                    error = f"exit code {proc.returncode}: {stderr_text[-500:]}"
                else:
                    error = "empty response"

            _save_raw_response(logs_dir, problem, stdout_text, stderr_text, success)

            return RunResult(
                problem_n=problem.n,
                problem_id=problem.problem_id,
                slug=problem.slug,
                raw_subject=problem.raw_subject,
                success=success,
                response_text=response_text,
                answer_code=answer_code,
                formal_eval=formal_eval,
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
                slug=problem.slug,
                raw_subject=problem.raw_subject,
                success=False,
                response_text=None,
                answer_code=None,
                formal_eval=None,
                error=f"timeout after {timeout:.0f}s",
                duration_s=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return RunResult(
                problem_n=problem.n,
                problem_id=problem.problem_id,
                slug=problem.slug,
                raw_subject=problem.raw_subject,
                success=False,
                response_text=None,
                answer_code=None,
                formal_eval=None,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=elapsed,
            )


# ---------------------------------------------------------------------------
# Output / metadata / logging
# ---------------------------------------------------------------------------


def _save_raw_response(
    logs_dir: Path | None,
    problem: Problem,
    stdout_text: str,
    stderr_text: str,
    success: bool,
) -> None:
    if logs_dir is None:
        return
    prefix = "ok" if success else "FAIL"
    log_path = logs_dir / f"{prefix}_{problem.n:03d}_{problem.slug}.txt"
    try:
        with open(log_path, "w") as f:
            f.write(f"=== STDOUT ({len(stdout_text)} chars) ===\n")
            f.write(stdout_text)
            f.write(f"\n\n=== STDERR ({len(stderr_text)} chars) ===\n")
            f.write(stderr_text)
    except OSError:
        pass


def _submission_path(output_dir: Path, problem_n: int, slug: str) -> Path:
    return output_dir / f"{problem_n:03d}_{slug}.json"


def write_submission_json(
    result: RunResult,
    output_dir: Path,
    model_string: str,
    generation_config: dict,
) -> Path | None:
    submission = {
        "problem_id": result.problem_id,
        "problem_n": result.problem_n,
        "slug": result.slug,
        "raw_subject": result.raw_subject,
        "source": "cais/hle",
        "model": model_string,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation_config": generation_config,
        "response_text": result.response_text,
        "generated_code": result.answer_code,
        "formal_eval": result.formal_eval,
        "stats": result.stats or None,
        "error": result.error,
        "duration_s": round(result.duration_s, 1),
    }
    out_path = _submission_path(output_dir, result.problem_n, result.slug)
    out_path.write_text(json.dumps(submission, indent=2, ensure_ascii=False))
    return out_path


def find_completed_submissions(output_dir: Path) -> set[int]:
    """Return problem indices that already have a non-empty submission JSON."""
    completed: set[int] = set()
    pat = re.compile(r"^(\d+)_.*\.json$")
    for f in output_dir.glob("*.json"):
        if f.name == "batch_metadata.json":
            continue
        m = pat.match(f.name)
        if not m:
            continue
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("problem_id") and data.get("response_text"):
            completed.add(int(m.group(1)))
    return completed


def make_output_dir(args: argparse.Namespace, create: bool = True) -> Path:
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = args.model.replace("/", "-").replace(":", "-")
        output_dir = DEFAULT_RESULTS_BASE / safe_model / ts
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_resume_config(resume_dir: Path) -> tuple[dict, dict]:
    meta_path = resume_dir / "batch_metadata.json"
    if not meta_path.exists():
        print(f"Error: no batch_metadata.json found in {resume_dir}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(meta_path.read_text())
    return data.get("generation_config", {}), data.get("run_config", {})


def write_batch_metadata(
    output_dir: Path,
    model_string: str,
    all_results: list[RunResult],
    generation_config: dict,
    run_config: dict,
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Write batch_metadata.json for the current batch (atomic, replace-on-write).

    Resume cycles overwrite the prior metadata; per-problem JSONs are the
    authoritative record. We intentionally don't carry forward previous_attempts
    history here — HLE batches are exploratory, not submission-shaped.
    """
    timestamp_iso = end_time.isoformat()

    entries: list[dict] = []
    for r in all_results:
        entry = {
            "problem_id": r.problem_id,
            "problem_n": r.problem_n,
            "slug": r.slug,
            "raw_subject": r.raw_subject,
            "success": r.success,
            "duration_s": round(r.duration_s, 1),
            "error": r.error,
            "formal_eval": r.formal_eval,
            "timestamp": timestamp_iso,
        }
        if r.stats:
            entry["stats"] = r.stats
        entries.append(entry)
    entries.sort(key=lambda e: e["problem_n"])

    # All on-disk submission JSONs (counts any prior-run output too).
    submission_ids: list[str] = []
    for f in sorted(output_dir.glob("*.json")):
        if f.name == "batch_metadata.json":
            continue
        try:
            data = json.loads(f.read_text())
            if data.get("problem_id"):
                submission_ids.append(data["problem_id"])
        except (json.JSONDecodeError, OSError):
            continue

    n_run_success = sum(1 for r in all_results if r.success)
    n_correct = sum(
        1
        for r in all_results
        if r.formal_eval and r.formal_eval.get("verdict") == "correct"
    )
    n_incorrect = sum(
        1
        for r in all_results
        if r.formal_eval and r.formal_eval.get("verdict") == "incorrect"
    )
    n_skipped = sum(
        1
        for r in all_results
        if r.formal_eval and r.formal_eval.get("verdict") == "skipped"
    )
    n_error = sum(
        1
        for r in all_results
        if r.formal_eval and r.formal_eval.get("verdict") == "error"
    )

    total_cost = sum(float(r.stats.get("cost_usd", 0) or 0) for r in all_results)
    total_input = sum(int(r.stats.get("input_tokens", 0) or 0) for r in all_results)
    total_output = sum(int(r.stats.get("output_tokens", 0) or 0) for r in all_results)
    total_duration = sum(float(r.duration_s or 0) for r in all_results)

    summary: dict = {
        "total_submissions": len(submission_ids),
        "this_run_total": len(all_results),
        "this_run_success": n_run_success,
        "this_run_failed": len(all_results) - n_run_success,
        "eval_correct": n_correct,
        "eval_incorrect": n_incorrect,
        "eval_skipped": n_skipped,
        "eval_error": n_error,
        "total_compute_s": round(total_duration, 1),
        "wall_clock_s": round((end_time - start_time).total_seconds(), 1),
    }
    if total_cost > 0:
        summary["total_cost_usd"] = round(total_cost, 4)
    if total_input > 0:
        summary["total_input_tokens"] = total_input
    if total_output > 0:
        summary["total_output_tokens"] = total_output

    metadata = {
        "model": model_string,
        "timestamp": timestamp_iso,
        "generation_config": generation_config,
        "run_config": run_config,
        "num_submissions": len(submission_ids),
        "problem_ids": submission_ids,
        "summary": summary,
        "problems": entries,
    }

    final_path = output_dir / "batch_metadata.json"
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    os.replace(tmp_path, final_path)


def write_initial_batch_metadata(
    output_dir: Path,
    model_string: str,
    generation_config: dict,
    run_config: dict,
    start_time: datetime,
) -> None:
    """Stub metadata before any workers spawn; makes a killed run resumable."""
    write_batch_metadata(
        output_dir,
        model_string,
        [],
        generation_config,
        run_config,
        start_time,
        start_time,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def setup_signal_handler(loop, tasks: list) -> None:
    cancelled = False

    def _handler():
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            print("\nInterrupted — cancelling pending tasks...", file=sys.stderr)
            for t in tasks:
                if not t.done():
                    t.cancel()

    loop.add_signal_handler(signal.SIGINT, _handler)


def print_final_summary(
    all_results: list[RunResult],
    total: int,
    succeeded: int,
    failed: int,
    start_time: datetime,
    end_time: datetime,
    output_dir: Path,
) -> None:
    wall_clock = (end_time - start_time).total_seconds()
    total_cost = sum(float(r.stats.get("cost_usd", 0) or 0) for r in all_results)
    n_correct = sum(
        1
        for r in all_results
        if r.formal_eval and r.formal_eval.get("verdict") == "correct"
    )
    n_incorrect = sum(
        1
        for r in all_results
        if r.formal_eval and r.formal_eval.get("verdict") == "incorrect"
    )
    n_skipped = sum(
        1
        for r in all_results
        if r.formal_eval and r.formal_eval.get("verdict") == "skipped"
    )

    cost_str = f", ${total_cost:.2f} est. cost" if total_cost > 0 else ""
    print("---", file=sys.stderr)
    print(
        f"Done: {succeeded}/{total} responses, {failed} failed "
        f"({wall_clock:.0f}s wall clock{cost_str})",
        file=sys.stderr,
    )
    print(
        f"Eval: {n_correct} correct, {n_incorrect} incorrect, {n_skipped} skipped",
        file=sys.stderr,
    )
    print(f"Output: {output_dir}", file=sys.stderr)

    if failed > 0:
        print("\nFailed problems:", file=sys.stderr)
        for r in sorted(all_results, key=lambda x: x.problem_n):
            if not r.success:
                print(f"  #{r.problem_n} ({r.slug}): {r.error}", file=sys.stderr)


async def run_batch(args: argparse.Namespace) -> int:
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
    model_string = resolve_provider_model_string(args.model)

    problems = discover_problems(args.problems_dir, args.problems)
    if not problems:
        print("Error: no problems found", file=sys.stderr)
        return 1

    output_dir = make_output_dir(args, create=not args.dry_run)
    logs_dir = output_dir / "logs"
    if not args.dry_run:
        logs_dir.mkdir(exist_ok=True)

    n_skip = 0
    if not args.force and not args.dry_run:
        completed = find_completed_submissions(output_dir)
        before = len(problems)
        problems = [p for p in problems if p.n not in completed]
        n_skip = before - len(problems)

    print(f"Model:       {args.model} ({model_string})", file=sys.stderr)
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
            print(f"  #{p.n:03d} {p.slug} ({p.problem_id})", file=sys.stderr)
        return 0

    generation_config = {
        "system": "physics_intern_hle_one_shot",
        "model_key": args.model,
        "use_python": False,
        "use_web_search": False,
        "parsing": False,
    }
    run_config = {
        "problems_dir": str(args.problems_dir),
        "problems_subset": args.problems,
    }

    semaphore = asyncio.Semaphore(args.concurrency)
    start_time = datetime.now(timezone.utc)
    write_initial_batch_metadata(
        output_dir, model_string, generation_config, run_config, start_time
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
            problem, args.model, args.timeout, semaphore, logs_dir
        )
        if result.response_text is not None:
            write_submission_json(result, output_dir, model_string, generation_config)

        async with lock:
            completed_count += 1
            if result.success:
                succeeded += 1
            else:
                failed += 1
            all_results.append(result)

            verdict = ""
            if result.formal_eval:
                v = result.formal_eval.get("verdict", "")
                verdict = f" eval={v}"
            status = "OK" if result.success else f"FAIL: {result.error}"
            cost_str = ""
            if result.stats and result.stats.get("cost_usd"):
                cost_str = f", ${result.stats['cost_usd']:.4f}"
            print(
                f"[{completed_count}/{total}] #{result.problem_n:03d} "
                f"{result.slug} ({result.duration_s:.0f}s{cost_str}){verdict} {status}",
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
                        slug=p.slug,
                        raw_subject=p.raw_subject,
                        success=False,
                        response_text=None,
                        answer_code=None,
                        formal_eval=None,
                        error=str(r),
                        duration_s=0,
                    )
                )
                failed += 1
                completed_count += 1

    end_time = datetime.now(timezone.utc)
    write_batch_metadata(
        output_dir,
        model_string,
        all_results,
        generation_config,
        run_config,
        start_time,
        end_time,
    )
    print_final_summary(
        all_results, total, succeeded, failed, start_time, end_time, output_dir
    )
    return 0 if failed == 0 else 1


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(run_batch(args)))


if __name__ == "__main__":
    main()
