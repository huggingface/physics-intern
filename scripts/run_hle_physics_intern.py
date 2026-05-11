#!/usr/bin/env python3
"""Run HLE-physics problems through PhysicsIntern with rolling parallelism.

Each problem = one full multi-agent engine run via ``physics_intern.main``.
Produces an HLE-native per-problem JSON progressively. Supports resume from
interrupted runs (both at the problem level and mid-run via --resume).

Usage:
    uv run python scripts/run_hle_physics_intern.py
    uv run python scripts/run_hle_physics_intern.py --model claude-4.7-opus --concurrency 5
    uv run python scripts/run_hle_physics_intern.py --problems 1-10 --config config.cluster.yaml
    uv run python scripts/run_hle_physics_intern.py --resume results/hle/<model>/<ts>/
    uv run python scripts/run_hle_physics_intern.py --dry-run

Engine-side parameters (max_iterations, max_wall_seconds,
max_total_output_tokens, max_cost_usd) live in --config; defaults come
from src/physics_intern/config.default.yaml.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from run_hle_common import (
    DEFAULT_PROBLEMS_DIR,
    DEFAULTS,
    PROJECT_ROOT,
    Problem,
    discover_problems,
    find_completed_submissions,
    resolve_model,
    resolve_provider_model_string,
    setup_signal_handler,
    submission_path,
)

from physics_intern.core.config import load_config_yaml  # noqa: E402
from physics_intern.utils.markdown import parse_frontmatter  # noqa: E402
from physics_intern.verification.evaluate import extract_answer_code  # noqa: E402

DEFAULT_WORKSPACE_BASE = PROJECT_ROOT / "workspaces"
DEFAULT_RESULTS_BASE = PROJECT_ROOT / "results" / "hle"
FORMATTER_REJECTION_PREFIX = "FORMATTER_REJECTION"

# Workspace suffixes used by sibling baselines — these must not be matched as
# multi-agent workspaces during resume planning. Includes both the CritPt
# baselines and the HLE one-shot suffix.
_NON_AGENT_SUFFIXES = ("_oneshot", "_rsa", "_autophysicist", "_hle_oneshot")


# ---------------------------------------------------------------------------
# Engine config
# ---------------------------------------------------------------------------


def resolve_engine_params(config_path: Path | None) -> dict:
    """Merge config.default.yaml with the user's --config override."""
    merged = dict(DEFAULTS)
    if config_path is not None:
        merged.update(load_config_yaml(config_path))
    return merged


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
    answer_code: str | None
    answer_text: str | None  # raw ANSWER.md content
    formal_eval: dict | None
    error: str | None
    duration_s: float
    returncode: int | None = None
    workspace_dir: Path | None = None
    soft_exit_reason: str | None = None
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run HLE-physics problems through PhysicsIntern in parallel.",
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
        "--config",
        type=Path,
        default=None,
        help=(
            "Config YAML file passed through to each engine subprocess. "
            "Engine-side parameters (max_iterations, max_wall_seconds, "
            "max_total_output_tokens, max_cost_usd, ...) are read from this file "
            "(merged on top of config.default.yaml)."
        ),
    )
    p.add_argument(
        "--concurrency", type=int, default=64, help="Max parallel runs (default: 64)"
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
        "--workspace-base",
        type=Path,
        default=DEFAULT_WORKSPACE_BASE,
        help="Base directory for workspaces",
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
        "--fresh",
        action="store_true",
        help="Ignore existing workspaces; start every problem from scratch",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    return p


# ---------------------------------------------------------------------------
# Workspace-based resume planning
# ---------------------------------------------------------------------------


def find_existing_workspace(
    slug: str,
    model_key: str,
    workspace_base: Path,
) -> Path | None:
    """Return the most recent *multi-agent* workspace for a problem slug."""
    safe_model = model_key.replace("/", "-").replace(":", "-")
    if not workspace_base.exists():
        return None
    matches: list[Path] = []
    for d in workspace_base.iterdir():
        if not d.is_dir():
            continue
        if slug not in d.name or safe_model not in d.name:
            continue
        if d.name.endswith(_NON_AGENT_SUFFIXES):
            continue
        matches.append(d)
    if not matches:
        return None
    matches.sort(key=lambda p: p.name, reverse=True)
    return matches[0]


@dataclass
class ResumeAction:
    problem: Problem
    action: str  # "skip", "extract", "resume", "fresh"
    workspace: Path | None = None
    answer_text: str | None = None
    answer_code: str | None = None
    cleanup_answer_before_resume: bool = False
    cleanup_answer_reason: str = ""


def _is_formatter_rejection_answer(answer_text: str) -> bool:
    return answer_text.strip().startswith(FORMATTER_REJECTION_PREFIX)


def _commit_answer_cleanup_before_resume(workspace: Path, reason: str) -> None:
    """Remove an invalid final answer and commit that cleanup before resume.

    ``physics_intern.main --resume`` refuses any workspace with ``ANSWER.md``
    because that file is the canonical completion signal. Empty or explicit
    formatter-rejection answers are not valid submissions, so the batch runner
    removes them and records the cleanup in the workspace git history before
    granting more budget.
    """
    answer_path = workspace / "ANSWER.md"
    if not answer_path.exists():
        return

    answer_text = answer_path.read_text()
    if answer_text.strip() and not _is_formatter_rejection_answer(answer_text):
        return

    answer_path.unlink()
    add_proc = subprocess.run(
        ["git", "add", "-A", "ANSWER.md"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=False,
    )
    if add_proc.returncode != 0:
        raise RuntimeError(
            f"Failed to stage ANSWER.md cleanup in {workspace}: "
            f"{add_proc.stderr.strip()}"
        )

    status_proc = subprocess.run(
        ["git", "status", "--porcelain", "--", "ANSWER.md"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=False,
    )
    if status_proc.returncode != 0:
        raise RuntimeError(
            f"Failed to inspect ANSWER.md cleanup in {workspace}: "
            f"{status_proc.stderr.strip()}"
        )
    if not status_proc.stdout.strip():
        return

    commit_proc = subprocess.run(
        ["git", "commit", "-m", f"Remove invalid ANSWER.md before resume: {reason}"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_proc.returncode != 0:
        raise RuntimeError(
            f"Failed to commit ANSWER.md cleanup in {workspace}: "
            f"{commit_proc.stderr.strip()}"
        )


def plan_actions(
    problems: list[Problem],
    output_dir: Path,
    workspace_base: Path,
    model_key: str,
    force: bool,
    fresh: bool = False,
) -> list[ResumeAction]:
    """Determine the action for each problem."""
    completed = (
        set()
        if force
        else find_completed_submissions(
            output_dir, required_keys=("problem_id", "answer_code")
        )
    )
    actions: list[ResumeAction] = []
    for p in problems:
        if p.n in completed:
            actions.append(ResumeAction(problem=p, action="skip"))
            continue

        ws = (
            None
            if fresh
            else find_existing_workspace(p.slug, model_key, workspace_base)
        )
        if ws:
            answer_path = ws / "ANSWER.md"
            cleanup_answer_before_resume = False
            cleanup_answer_reason = ""
            if answer_path.exists():
                text = answer_path.read_text()
                stripped = text.strip()
                if stripped and not _is_formatter_rejection_answer(stripped):
                    code = extract_answer_code(text) or stripped
                    actions.append(
                        ResumeAction(
                            problem=p,
                            action="extract",
                            workspace=ws,
                            answer_text=text,
                            answer_code=code,
                        )
                    )
                    continue
                cleanup_answer_before_resume = True
                cleanup_answer_reason = (
                    "formatter rejection"
                    if _is_formatter_rejection_answer(stripped)
                    else "empty answer"
                )
            graph_path = ws / "RESEARCH_GRAPH.json"
            if graph_path.exists():
                actions.append(
                    ResumeAction(
                        problem=p,
                        action="resume",
                        workspace=ws,
                        cleanup_answer_before_resume=cleanup_answer_before_resume,
                        cleanup_answer_reason=cleanup_answer_reason,
                    )
                )
                continue
        actions.append(ResumeAction(problem=p, action="fresh"))
    return actions


# ---------------------------------------------------------------------------
# Live progress (per-iteration line + API/stall warnings)
# ---------------------------------------------------------------------------

_ITERATION_RE = re.compile(r"ITERATION\s+(\d+)")
_RETRY_RE = re.compile(r"Transient API error \(attempt (\d+)/(\d+)\)(?::\s*(.+))?")

_STALL_WARN_AFTER_S = 15 * 60
_STALL_REWARN_EVERY_S = 30 * 60
_WATCHDOG_TICK_S = 60

_running: dict[str, dict] = {}


def _exc_label(tail: str | None) -> str:
    if not tail:
        return "unknown"
    tail = tail.strip()
    if tail.startswith("Error code:"):
        m = re.match(r"Error code:\s*(\d+)", tail)
        if m:
            return f"HTTP {m.group(1)}"
    lower = tail.lower()
    if "timed out" in lower or "timeout" in lower:
        return "Timeout"
    if "connection" in lower:
        return "ConnectionError"
    m = re.match(r"([A-Z][A-Za-z0-9_]*Error)", tail)
    if m:
        return m.group(1)
    return tail[:40]


def _read_soft_exit_reason(workspace_dir: Path) -> str | None:
    """Detect whether the most recent run ended via the forced formatter."""
    state_path = workspace_dir / "RESEARCH_GRAPH.json"
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("status") != "partially_complete":
        return None
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--grep=forced formatter", "--pretty=%s"],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except (OSError, ValueError):
        return "unknown"
    m = re.search(r"forced formatter \(([^)]+)\)", out)
    return m.group(1) if m else "unknown"


def _read_formal_eval(workspace_dir: Path) -> dict | None:
    """Parse VERIFICATION.md frontmatter to recover the formal_answer verdict."""
    report_path = workspace_dir / "VERIFICATION.md"
    if not report_path.exists():
        return None
    try:
        content = report_path.read_text()
    except OSError:
        return None
    fm, _ = parse_frontmatter(content)
    formal_answer = fm.get("formal_answer")
    if not formal_answer:
        return None
    return {"verdict": str(formal_answer), "method": "from_report", "error": None}


async def _stall_watchdog(print_lock: asyncio.Lock) -> None:
    try:
        while True:
            await asyncio.sleep(_WATCHDOG_TICK_S)
            now = time.monotonic()
            to_warn: list[tuple[int, str, int]] = []
            for state in list(_running.values()):
                silent_s = now - state["last_line_at"]
                if silent_s < _STALL_WARN_AFTER_S:
                    continue
                last_warn = state.get("last_stall_warn_at")
                if last_warn is not None and (now - last_warn) < _STALL_REWARN_EVERY_S:
                    continue
                state["last_stall_warn_at"] = now
                to_warn.append((state["problem_n"], state["slug"], int(silent_s // 60)))
            if to_warn:
                async with print_lock:
                    for pn, slug, mins in to_warn:
                        print(
                            f"  #{pn:03d} {slug}   ⚠ stalled {mins}m, no output",
                            file=sys.stderr,
                        )
    except asyncio.CancelledError:
        return


# ---------------------------------------------------------------------------
# Worker: run one problem
# ---------------------------------------------------------------------------


async def run_one_problem(
    action: ResumeAction,
    model_key: str,
    config_path: Path | None,
    workspace_base: Path,
    semaphore: asyncio.Semaphore,
    print_lock: asyncio.Lock,
) -> RunResult:
    """Run a single HLE problem as a ``physics_intern.main`` subprocess."""
    problem = action.problem

    if action.action == "extract":
        return RunResult(
            problem_n=problem.n,
            problem_id=problem.problem_id,
            slug=problem.slug,
            raw_subject=problem.raw_subject,
            success=True,
            answer_code=action.answer_code,
            answer_text=action.answer_text,
            formal_eval=_read_formal_eval(action.workspace)
            if action.workspace
            else None,
            error=None,
            duration_s=0.0,
            workspace_dir=action.workspace,
            soft_exit_reason=_read_soft_exit_reason(action.workspace)
            if action.workspace
            else None,
        )

    async with semaphore:
        if action.action == "resume" and action.workspace:
            ws = action.workspace
            p = await asyncio.create_subprocess_exec(
                "git",
                "checkout",
                ".",
                cwd=str(ws),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await p.wait()
            p = await asyncio.create_subprocess_exec(
                "git",
                "clean",
                "-fd",
                cwd=str(ws),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await p.wait()
            if action.cleanup_answer_before_resume:
                _commit_answer_cleanup_before_resume(ws, action.cleanup_answer_reason)
            cmd = [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-m",
                "physics_intern.main",
                "--resume",
                str(action.workspace),
            ]
            workspace_dir = action.workspace
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model = model_key.replace("/", "-").replace(":", "-")
            ws_name = f"{timestamp}_{problem.slug}_{safe_model}"
            workspace_dir = workspace_base / ws_name

            cmd = [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-m",
                "physics_intern.main",
                str(problem.yaml_path),
                "--model",
                model_key,
                "--workspace-dir",
                str(workspace_dir),
            ]

        if config_path:
            cmd.extend(["--config", str(config_path)])

        start = time.monotonic()
        state = {
            "problem_n": problem.n,
            "slug": problem.slug,
            "start_at": start,
            "last_line_at": start,
            "iter": 0,
            "api_retries": 0,
            "last_stall_warn_at": None,
        }
        _running[problem.problem_id] = state
        stats = {"api_retries": 0}
        stderr_tail: list[str] = []
        proc = None
        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
                start_new_session=True,
                env=env,
            )

            async def _stream_stdout():
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    text = line.decode(errors="replace")
                    now = time.monotonic()
                    state["last_line_at"] = now
                    state["last_stall_warn_at"] = None

                    m = _ITERATION_RE.search(text)
                    if m:
                        state["iter"] = int(m.group(1))
                        elapsed_so_far = now - start
                        async with print_lock:
                            print(
                                f"  #{problem.n:03d} {problem.slug}   "
                                f"iter {m.group(1)}   ({elapsed_so_far:.0f}s)",
                                file=sys.stderr,
                            )
                        continue

                    m = _RETRY_RE.search(text)
                    if m:
                        state["api_retries"] += 1
                        attempt = int(m.group(1))
                        max_att = int(m.group(2))
                        exc_label = _exc_label(m.group(3))
                        if attempt >= 2:
                            async with print_lock:
                                print(
                                    f"  #{problem.n:03d} {problem.slug}   ⚠ API  "
                                    f"attempt {attempt}/{max_att}  ({exc_label})",
                                    file=sys.stderr,
                                )
                        continue

            async def _drain_stderr():
                assert proc.stderr is not None
                data = await proc.stderr.read()
                text = data.decode(errors="replace")
                for line in text.splitlines():
                    stderr_tail.append(line)
                if len(stderr_tail) > 50:
                    del stderr_tail[:-50]

            await asyncio.gather(_stream_stdout(), _drain_stderr(), proc.wait())
            elapsed = time.monotonic() - start
            stats["api_retries"] = state["api_retries"]

            answer_text: str | None = None
            answer_code: str | None = None
            answer_path = workspace_dir / "ANSWER.md"
            if answer_path.exists():
                raw = answer_path.read_text()
                stripped = raw.strip()
                if stripped and not stripped.startswith(FORMATTER_REJECTION_PREFIX):
                    answer_text = raw
                    answer_code = extract_answer_code(raw) or stripped

            formal_eval = _read_formal_eval(workspace_dir)
            soft_exit = _read_soft_exit_reason(workspace_dir)

            success = answer_code is not None
            error: str | None = None
            if proc.returncode != 0 and answer_code is None:
                tail = "".join(stderr_tail)[-500:]
                error = f"exit code {proc.returncode}: {tail}"
            elif answer_code is None:
                error = "no valid ANSWER.md produced"

            return RunResult(
                problem_n=problem.n,
                problem_id=problem.problem_id,
                slug=problem.slug,
                raw_subject=problem.raw_subject,
                success=success,
                answer_code=answer_code,
                answer_text=answer_text,
                formal_eval=formal_eval,
                error=error,
                duration_s=elapsed,
                returncode=proc.returncode,
                workspace_dir=workspace_dir,
                soft_exit_reason=soft_exit,
                stats=stats,
            )

        except Exception as exc:
            elapsed = time.monotonic() - start
            stats["api_retries"] = state["api_retries"]
            return RunResult(
                problem_n=problem.n,
                problem_id=problem.problem_id,
                slug=problem.slug,
                raw_subject=problem.raw_subject,
                success=False,
                answer_code=None,
                answer_text=None,
                formal_eval=None,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=elapsed,
                workspace_dir=workspace_dir,
                stats=stats,
            )
        finally:
            _running.pop(problem.problem_id, None)


# ---------------------------------------------------------------------------
# Output / metadata
# ---------------------------------------------------------------------------


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
        "answer_text": result.answer_text,
        "generated_code": result.answer_code,
        "answer_code": result.answer_code,  # alias used by find_completed_submissions
        "formal_eval": result.formal_eval,
        "stats": result.stats or None,
        "workspace_dir": str(result.workspace_dir) if result.workspace_dir else None,
        "soft_exit_reason": result.soft_exit_reason,
        "error": result.error,
        "duration_s": round(result.duration_s, 1),
    }
    out_path = submission_path(output_dir, result.problem_n, result.slug)
    out_path.write_text(json.dumps(submission, indent=2, ensure_ascii=False))
    return out_path


def load_resume_config(resume_dir: Path) -> tuple[dict, dict]:
    meta_path = resume_dir / "batch_metadata.json"
    if not meta_path.exists():
        print(f"Error: no batch_metadata.json found in {resume_dir}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(meta_path.read_text())
    return data.get("generation_config", {}), data.get("run_config", {})


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


def _eval_buckets(results: list[RunResult]) -> dict[str, int]:
    """Tally formal_eval verdicts (correct/incorrect/skipped/inconclusive)."""
    buckets: dict[str, int] = {}
    for r in results:
        v = (r.formal_eval or {}).get("verdict")
        if not v:
            continue
        buckets[v] = buckets.get(v, 0) + 1
    return buckets


def write_batch_metadata(
    output_dir: Path,
    model_string: str,
    all_results: list[RunResult],
    generation_config: dict,
    run_config: dict,
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Write batch_metadata.json (atomic). Per-problem JSONs remain authoritative."""
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
            "workspace_dir": str(r.workspace_dir) if r.workspace_dir else None,
            "soft_exit_reason": r.soft_exit_reason,
            "timestamp": timestamp_iso,
        }
        if r.stats:
            entry["stats"] = r.stats
        entries.append(entry)
    entries.sort(key=lambda e: e["problem_n"])

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
    buckets = _eval_buckets(all_results)
    total_duration = sum(float(r.duration_s or 0) for r in all_results)

    summary: dict = {
        "total_submissions": len(submission_ids),
        "this_run_total": len(all_results),
        "this_run_success": n_run_success,
        "this_run_failed": len(all_results) - n_run_success,
        "eval_correct": buckets.get("correct", 0),
        "eval_incorrect": buckets.get("incorrect", 0),
        "eval_skipped": buckets.get("skipped", 0),
        "eval_inconclusive": buckets.get("inconclusive", 0),
        "total_compute_s": round(total_duration, 1),
        "wall_clock_s": round((end_time - start_time).total_seconds(), 1),
    }

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
# Final summary
# ---------------------------------------------------------------------------


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
    buckets = _eval_buckets(all_results)

    print("---", file=sys.stderr)
    print(
        f"Done: {succeeded}/{total} answers, {failed} failed "
        f"({wall_clock:.0f}s wall clock)",
        file=sys.stderr,
    )
    print(
        f"Eval: {buckets.get('correct', 0)} correct, "
        f"{buckets.get('incorrect', 0)} incorrect, "
        f"{buckets.get('skipped', 0)} skipped, "
        f"{buckets.get('inconclusive', 0)} inconclusive",
        file=sys.stderr,
    )
    print(f"Output: {output_dir}", file=sys.stderr)

    if failed > 0:
        print("\nFailed problems:", file=sys.stderr)
        for r in sorted(all_results, key=lambda x: x.problem_n):
            if not r.success:
                print(f"  #{r.problem_n:03d} {r.slug}: {r.error}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run_batch(args: argparse.Namespace) -> int:
    if args.resume:
        if not args.resume.is_dir():
            print(f"Error: resume directory not found: {args.resume}", file=sys.stderr)
            return 1
        gen_cfg, run_cfg = load_resume_config(args.resume)
        args.output_dir = args.resume
        if args.model is None:
            args.model = gen_cfg.get("model_key")
        if args.config is None and gen_cfg.get("config_file"):
            args.config = Path(gen_cfg["config_file"])
        if run_cfg.get("problems_dir"):
            args.problems_dir = Path(run_cfg["problems_dir"])
        if run_cfg.get("problems_subset"):
            args.problems = run_cfg["problems_subset"]
        if run_cfg.get("workspace_base"):
            args.workspace_base = Path(run_cfg["workspace_base"])
        print(f"Resuming from {args.resume}", file=sys.stderr)

    eng = resolve_engine_params(args.config)
    resolve_model(args, args.output_dir, config_model=eng.get("model"))
    model_string = resolve_provider_model_string(args.model)

    problems = discover_problems(args.problems_dir, args.problems)
    if not problems:
        print("Error: no problems found", file=sys.stderr)
        return 1

    output_dir = make_output_dir(args, create=not args.dry_run)

    actions = plan_actions(
        problems,
        output_dir,
        args.workspace_base,
        args.model,
        args.force,
        fresh=args.fresh,
    )

    n_skip = sum(1 for a in actions if a.action == "skip")
    n_extract = sum(1 for a in actions if a.action == "extract")
    n_resume = sum(1 for a in actions if a.action == "resume")
    n_fresh = sum(1 for a in actions if a.action == "fresh")

    config_label = str(args.config) if args.config else "(defaults only)"
    wall_s = int(eng.get("max_wall_seconds", 0) or 0)
    wall_str = f"{wall_s}s" if wall_s > 0 else "disabled"
    tok_budget = int(eng.get("max_total_output_tokens", 0) or 0)
    tok_str = f"{tok_budget:,}" if tok_budget > 0 else "disabled"
    cost_budget = float(eng.get("max_cost_usd", 0.0) or 0.0)
    cost_str = f"${cost_budget:.2f}" if cost_budget > 0 else "disabled"

    print(f"Model:           {args.model} ({model_string})", file=sys.stderr)
    print(f"Config:          {config_label}", file=sys.stderr)
    print(f"Max iterations:  {eng['max_iterations']}", file=sys.stderr)
    print(f"Max wall time:   {wall_str}", file=sys.stderr)
    print(f"Max out tokens:  {tok_str}", file=sys.stderr)
    print(f"Max cost:        {cost_str}", file=sys.stderr)
    print(f"Problems:        {len(problems)} total", file=sys.stderr)
    print(f"  skip:          {n_skip} (submission exists)", file=sys.stderr)
    print(f"  extract:       {n_extract} (answer exists, write JSON)", file=sys.stderr)
    print(f"  resume:        {n_resume} (continue interrupted run)", file=sys.stderr)
    print(f"  fresh:         {n_fresh} (new run)", file=sys.stderr)
    print(f"Concurrency:     {args.concurrency}", file=sys.stderr)
    print(f"Output:          {output_dir}", file=sys.stderr)
    print("---", file=sys.stderr)

    to_run = [a for a in actions if a.action != "skip"]
    if not to_run:
        print("All problems already completed.", file=sys.stderr)
        return 0

    if args.dry_run:
        for a in to_run:
            ws_note = f"  (ws: {a.workspace.name})" if a.workspace else ""
            print(
                f"  [{a.action:7s}] #{a.problem.n:03d} {a.problem.slug}{ws_note}",
                file=sys.stderr,
            )
        return 0

    generation_config = {
        "system": "physics_intern",
        "model_key": args.model,
        "max_iterations": eng["max_iterations"],
        "max_wall_seconds": eng.get("max_wall_seconds", 0),
        "max_total_output_tokens": eng.get("max_total_output_tokens", 0),
        "max_cost_usd": eng.get("max_cost_usd", 0.0),
        "config_file": str(args.config) if args.config else None,
        "use_python": True,
        "use_web_search": False,
        "parsing": False,
    }
    run_config = {
        "problems_dir": str(args.problems_dir),
        "problems_subset": args.problems,
        "workspace_base": str(args.workspace_base),
    }

    semaphore = asyncio.Semaphore(args.concurrency)
    start_time = datetime.now(timezone.utc)
    write_initial_batch_metadata(
        output_dir, model_string, generation_config, run_config, start_time
    )

    total = len(to_run)
    completed = 0
    succeeded = 0
    failed = 0
    all_results: list[RunResult] = []
    lock = asyncio.Lock()
    print_lock = asyncio.Lock()

    async def worker(action: ResumeAction) -> RunResult:
        nonlocal completed, succeeded, failed

        result = await run_one_problem(
            action,
            args.model,
            args.config,
            args.workspace_base,
            semaphore,
            print_lock,
        )

        if result.success:
            write_submission_json(result, output_dir, model_string, generation_config)

        async with lock:
            completed += 1
            if result.success:
                succeeded += 1
            else:
                failed += 1
            all_results.append(result)

            verdict = ""
            if result.formal_eval:
                v = result.formal_eval.get("verdict", "")
                verdict = f" eval={v}"
            if result.success:
                if result.soft_exit_reason:
                    status = f"OK (soft-exit: {result.soft_exit_reason})"
                else:
                    status = "OK"
            else:
                status = f"FAIL: {result.error}"
            retries = (result.stats or {}).get("api_retries", 0) if result.stats else 0
            retry_note = f", {retries} API retries" if retries else ""
            if result.duration_s > 0:
                print(
                    f"[{completed}/{total}] #{result.problem_n:03d} {result.slug} "
                    f"({action.action}, {result.duration_s:.0f}s){verdict} "
                    f"{status}{retry_note}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[{completed}/{total}] #{result.problem_n:03d} {result.slug} "
                    f"({action.action}){verdict} {status}{retry_note}",
                    file=sys.stderr,
                )

        return result

    tasks = [asyncio.create_task(worker(a)) for a in to_run]
    watchdog = asyncio.create_task(_stall_watchdog(print_lock))

    loop = asyncio.get_running_loop()
    setup_signal_handler(loop, tasks)

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        watchdog.cancel()
        try:
            await watchdog
        except (asyncio.CancelledError, Exception):
            pass

    for i, r in enumerate(results):
        if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
            p = to_run[i].problem
            async with lock:
                all_results.append(
                    RunResult(
                        problem_n=p.n,
                        problem_id=p.problem_id,
                        slug=p.slug,
                        raw_subject=p.raw_subject,
                        success=False,
                        answer_code=None,
                        answer_text=None,
                        formal_eval=None,
                        error=str(r),
                        duration_s=0,
                    )
                )
                failed += 1
                completed += 1

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
