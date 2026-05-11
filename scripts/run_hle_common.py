"""Shared utilities for HLE batch runner scripts.

Mirrors the role of :mod:`run_critpt_common` for the HLE family. Provides
problem discovery, model resolution, and orchestration helpers that are
independent of the runner-specific output format.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = PROJECT_ROOT / "src" / "physics_intern" / "models.yaml"
DEFAULT_PROBLEMS_DIR = PROJECT_ROOT / "problems" / "hle-physics"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from physics_intern.core.config import DEFAULTS  # noqa: E402


# ---------------------------------------------------------------------------
# Problem discovery
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    n: int  # 1-based index in alphabetical filename order
    problem_id: str  # the YAML's `id` field (24-char HLE hex)
    slug: str  # filename stem
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


def resolve_model(
    args: argparse.Namespace,
    output_dir: Path | None,
    config_model: str | None = None,
) -> str:
    """Resolve model with precedence: --model > resumed > config > default."""
    if args.model is None and output_dir and output_dir.exists():
        recovered = _read_model_from_output_dir(output_dir)
        if recovered:
            args.model = recovered
            print(f"Resumed model from previous run: {recovered}", file=sys.stderr)
    if args.model is None and config_model is not None:
        args.model = config_model
    if args.model is None:
        args.model = DEFAULTS["model"]
    return args.model


# ---------------------------------------------------------------------------
# On-disk submission lookups (shared shape: ``NNN_<slug>.json``)
# ---------------------------------------------------------------------------


_SUBMISSION_FILENAME_RE = re.compile(r"^(\d+)_.*\.json$")


def find_completed_submissions(
    output_dir: Path,
    *,
    required_keys: tuple[str, ...] = ("problem_id",),
) -> set[int]:
    """Return problem indices whose submission JSON has all ``required_keys``.

    The HLE one-shot runner requires ``response_text`` on top of ``problem_id``;
    the multi-agent runner requires ``generated_code``. Callers pass the keys
    they need.
    """
    completed: set[int] = set()
    for f in output_dir.glob("*.json"):
        if f.name == "batch_metadata.json":
            continue
        m = _SUBMISSION_FILENAME_RE.match(f.name)
        if not m:
            continue
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if all(data.get(k) for k in required_keys):
            completed.add(int(m.group(1)))
    return completed


def submission_path(output_dir: Path, problem_n: int, slug: str) -> Path:
    """Canonical per-problem JSON path: ``<output_dir>/NNN_<slug>.json``."""
    return output_dir / f"{problem_n:03d}_{slug}.json"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def setup_signal_handler(loop, tasks: list) -> None:
    """Cancel pending asyncio tasks on Ctrl+C."""
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
