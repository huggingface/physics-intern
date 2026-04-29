#!/usr/bin/env python3
"""Analyze token usage and metrics across a CritPt batch run.

Reads METRICS.md files from workspaces referenced by a batch results directory
and produces aggregate statistics (total, mean, median, percentiles) for
token counts, iterations, and per-agent breakdowns.

Usage:
    uv run python scripts/analyze_batch.py results/critpt/gemini-3-flash-preview/20260407_094509
    uv run python scripts/analyze_batch.py results/critpt/gemini-3-flash-preview/20260407_094509 --json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from open_dirac.metrics import parse_metrics_table, parse_yaml_frontmatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE_BASE = PROJECT_ROOT / "workspaces"

console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ProblemMetrics:
    problem_id: str
    problem_n: int
    workspace: Path
    total_iterations: int = 0
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_answer_tokens: int = 0
    total_tool_calls: int = 0
    max_tokens_reached_count: int = 0
    num_sanity_checks: int = 0
    num_critiques: int = 0
    agent_stats: dict[str, AgentStats] = field(default_factory=dict)


@dataclass
class AgentStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Workspace discovery
# ---------------------------------------------------------------------------


def find_existing_workspace(
    problem_id: str,
    model_key: str,
    workspace_base: Path,
) -> Path | None:
    """Find the most recent workspace for a problem+model pair."""
    safe_model = model_key.replace("/", "-").replace(":", "-")
    matches: list[Path] = []
    if not workspace_base.exists():
        return None
    for d in workspace_base.iterdir():
        if d.is_dir() and problem_id in d.name and safe_model in d.name:
            matches.append(d)
    if not matches:
        return None
    matches.sort(key=lambda p: p.name, reverse=True)
    return matches[0]


# ---------------------------------------------------------------------------
# METRICS.md parsing
# ---------------------------------------------------------------------------


def load_problem_metrics(problem_id: str, workspace: Path) -> ProblemMetrics | None:
    """Load metrics from a workspace's METRICS.md."""
    metrics_path = workspace / "METRICS.md"
    if not metrics_path.exists():
        return None

    text = metrics_path.read_text()
    fm = parse_yaml_frontmatter(text)
    if not fm:
        return None

    n_match = re.search(r"Challenge_(\d+)_main", problem_id)
    problem_n = int(n_match.group(1)) if n_match else 0

    pm = ProblemMetrics(
        problem_id=problem_id,
        problem_n=problem_n,
        workspace=workspace,
        total_iterations=fm.get("total_iterations", 0),
        total_llm_calls=fm.get("total_llm_calls", 0),
        total_input_tokens=fm.get("total_input_tokens", 0),
        total_output_tokens=fm.get("total_output_tokens", 0),
        total_reasoning_tokens=fm.get("total_reasoning_tokens", 0),
        total_answer_tokens=fm.get("total_answer_tokens", 0),
        total_tool_calls=fm.get("total_tool_calls", 0),
        max_tokens_reached_count=fm.get("max_tokens_reached_count", 0),
    )

    # Counts from RESEARCH_GRAPH.json (source of truth for sanity checks & critiques)
    graph_path = workspace / "RESEARCH_GRAPH.json"
    if graph_path.exists():
        try:
            graph = json.loads(graph_path.read_text())
            pm.num_sanity_checks = len(graph.get("sanity_checks", []) or [])
            pm.num_critiques = len(graph.get("critiques", {}) or {})
        except (json.JSONDecodeError, OSError):
            pass

    # Parse per-iteration table for agent breakdown
    table_rows = parse_metrics_table(text)
    for row in table_rows:
        agent = row["agent"]
        if agent not in pm.agent_stats:
            pm.agent_stats[agent] = AgentStats()
        s = pm.agent_stats[agent]
        s.calls += 1
        s.input_tokens += row["input_tokens"]
        s.output_tokens += row["output_tokens"]
        s.tool_calls += row["tool_calls"]
        s.duration_s += row["duration_s"]

    return pm


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def compute_stats(values: list[int | float]) -> dict:
    """Compute summary statistics for a list of values."""
    if not values:
        return {
            "n": 0,
            "total": 0,
            "mean": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "std": 0,
            "p25": 0,
            "p75": 0,
        }
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    mean = total / n
    median = statistics.median(sorted_vals)
    std = statistics.stdev(sorted_vals) if n > 1 else 0.0
    p25 = sorted_vals[max(0, n // 4 - 1)] if n >= 4 else sorted_vals[0]
    p75 = sorted_vals[min(n - 1, 3 * n // 4)] if n >= 4 else sorted_vals[-1]
    return {
        "n": n,
        "total": total,
        "mean": round(mean, 1),
        "median": median,
        "min": min(sorted_vals),
        "max": max(sorted_vals),
        "std": round(std, 1),
        "p25": p25,
        "p75": p75,
    }


def fmt_tokens(n: int | float) -> str:
    """Format token count with K/M suffix."""
    if isinstance(n, float):
        n = round(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_summary(
    all_metrics: list[ProblemMetrics],
    model: str,
    results_dir: Path,
) -> None:
    """Print rich summary tables to stderr."""
    n = len(all_metrics)

    # Header
    total_input = sum(m.total_input_tokens for m in all_metrics)
    total_output = sum(m.total_output_tokens for m in all_metrics)
    total_reasoning = sum(m.total_reasoning_tokens for m in all_metrics)
    total_answer = sum(m.total_answer_tokens for m in all_metrics)
    total_all = total_input + total_output

    console.print()
    console.print(f"[bold]Batch Analysis: {results_dir.name}[/bold]")
    console.print(f"Model: {model}")
    console.print(f"Problems: {n}")
    console.print(
        f"Total tokens: {fmt_tokens(total_all)} "
        f"(input: {fmt_tokens(total_input)}, output: {fmt_tokens(total_output)})"
    )
    console.print(
        f"  reasoning: {fmt_tokens(total_reasoning)}, "
        f"answer: {fmt_tokens(total_answer)}"
    )
    console.print()

    # Distribution table
    metrics_map = {
        "Iterations": [m.total_iterations for m in all_metrics],
        "LLM calls": [m.total_llm_calls for m in all_metrics],
        "Input tokens": [m.total_input_tokens for m in all_metrics],
        "Output tokens": [m.total_output_tokens for m in all_metrics],
        "Reasoning tokens": [m.total_reasoning_tokens for m in all_metrics],
        "Answer tokens": [m.total_answer_tokens for m in all_metrics],
        "Tool calls": [m.total_tool_calls for m in all_metrics],
        "Sanity checks": [m.num_sanity_checks for m in all_metrics],
        "Critiques": [m.num_critiques for m in all_metrics],
    }

    table = Table(title="Distribution across problems")
    table.add_column("Metric", style="bold")
    for col in ["Total", "Mean", "Median", "Min", "Max", "Std", "P25", "P75"]:
        table.add_column(col, justify="right")

    for name, values in metrics_map.items():
        s = compute_stats(values)
        is_tokens = "tokens" in name.lower()
        fmt = (
            fmt_tokens
            if is_tokens
            else lambda x: f"{x:,.0f}" if isinstance(x, float) else f"{x:,}"
        )
        table.add_row(
            name,
            fmt(s["total"]),
            fmt(s["mean"]),
            fmt(s["median"]),
            fmt(s["min"]),
            fmt(s["max"]),
            fmt(s["std"]),
            fmt(s["p25"]),
            fmt(s["p75"]),
        )

    console.print(table)
    console.print()

    # Per-agent breakdown
    agent_totals: dict[str, AgentStats] = {}
    for m in all_metrics:
        for agent_name, stats in m.agent_stats.items():
            if agent_name not in agent_totals:
                agent_totals[agent_name] = AgentStats()
            t = agent_totals[agent_name]
            t.calls += stats.calls
            t.input_tokens += stats.input_tokens
            t.output_tokens += stats.output_tokens
            t.tool_calls += stats.tool_calls
            t.duration_s += stats.duration_s

    if agent_totals:
        agent_table = Table(title="Per-agent totals")
        agent_table.add_column("Agent", style="bold")
        agent_table.add_column("Calls", justify="right")
        agent_table.add_column("Input tokens", justify="right")
        agent_table.add_column("Output tokens", justify="right")
        agent_table.add_column("Tool calls", justify="right")
        agent_table.add_column("Duration", justify="right")
        agent_table.add_column("% output", justify="right")

        total_out = sum(a.output_tokens for a in agent_totals.values())
        for name in sorted(
            agent_totals, key=lambda k: agent_totals[k].output_tokens, reverse=True
        ):
            a = agent_totals[name]
            pct = (a.output_tokens / total_out * 100) if total_out else 0
            hours = a.duration_s / 3600
            agent_table.add_row(
                name,
                f"{a.calls:,}",
                fmt_tokens(a.input_tokens),
                fmt_tokens(a.output_tokens),
                f"{a.tool_calls:,}",
                f"{hours:.1f}h",
                f"{pct:.1f}%",
            )

        console.print(agent_table)
        console.print()

    # Top/bottom 5 by total tokens
    sorted_by_tokens = sorted(
        all_metrics, key=lambda m: m.total_input_tokens + m.total_output_tokens
    )
    extremes_table = Table(title="Extremes by total tokens (input + output)")
    extremes_table.add_column("Problem", style="bold")
    extremes_table.add_column("Total tokens", justify="right")
    extremes_table.add_column("Iterations", justify="right")
    extremes_table.add_column("LLM calls", justify="right")

    extremes_table.add_row("[dim]--- Bottom 5 ---[/dim]", "", "", "")
    for m in sorted_by_tokens[:5]:
        extremes_table.add_row(
            f"C{m.problem_n}",
            fmt_tokens(m.total_input_tokens + m.total_output_tokens),
            str(m.total_iterations),
            str(m.total_llm_calls),
        )
    extremes_table.add_row("[dim]--- Top 5 ---[/dim]", "", "", "")
    for m in sorted_by_tokens[-5:]:
        extremes_table.add_row(
            f"C{m.problem_n}",
            fmt_tokens(m.total_input_tokens + m.total_output_tokens),
            str(m.total_iterations),
            str(m.total_llm_calls),
        )

    console.print(extremes_table)
    console.print()


def build_json_output(
    all_metrics: list[ProblemMetrics],
    model: str,
    results_dir: Path,
) -> dict:
    """Build a JSON-serializable summary."""
    metrics_map = {
        "iterations": [m.total_iterations for m in all_metrics],
        "llm_calls": [m.total_llm_calls for m in all_metrics],
        "input_tokens": [m.total_input_tokens for m in all_metrics],
        "output_tokens": [m.total_output_tokens for m in all_metrics],
        "reasoning_tokens": [m.total_reasoning_tokens for m in all_metrics],
        "answer_tokens": [m.total_answer_tokens for m in all_metrics],
        "tool_calls": [m.total_tool_calls for m in all_metrics],
        "sanity_checks": [m.num_sanity_checks for m in all_metrics],
        "critiques": [m.num_critiques for m in all_metrics],
    }

    agent_totals: dict[str, dict] = {}
    for m in all_metrics:
        for agent_name, stats in m.agent_stats.items():
            if agent_name not in agent_totals:
                agent_totals[agent_name] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "tool_calls": 0,
                    "duration_s": 0.0,
                }
            t = agent_totals[agent_name]
            t["calls"] += stats.calls
            t["input_tokens"] += stats.input_tokens
            t["output_tokens"] += stats.output_tokens
            t["tool_calls"] += stats.tool_calls
            t["duration_s"] += round(stats.duration_s, 1)

    return {
        "results_dir": str(results_dir),
        "model": model,
        "num_problems": len(all_metrics),
        "distributions": {k: compute_stats(v) for k, v in metrics_map.items()},
        "agent_totals": agent_totals,
        "per_problem": [
            {
                "problem_id": m.problem_id,
                "problem_n": m.problem_n,
                "total_iterations": m.total_iterations,
                "total_llm_calls": m.total_llm_calls,
                "total_input_tokens": m.total_input_tokens,
                "total_output_tokens": m.total_output_tokens,
                "total_reasoning_tokens": m.total_reasoning_tokens,
                "total_answer_tokens": m.total_answer_tokens,
                "total_tool_calls": m.total_tool_calls,
                "num_sanity_checks": m.num_sanity_checks,
                "num_critiques": m.num_critiques,
            }
            for m in sorted(all_metrics, key=lambda m: m.problem_n)
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze token usage across a CritPt batch run.",
    )
    parser.add_argument(
        "results_dir", type=Path, help="Path to batch results directory"
    )
    parser.add_argument(
        "--workspace-base",
        type=Path,
        default=DEFAULT_WORKSPACE_BASE,
        help="Base directory for workspaces",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output JSON summary to stdout"
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    metadata_path = results_dir / "batch_metadata.json"
    if not metadata_path.exists():
        console.print(f"[red]Error: {metadata_path} not found[/red]")
        return 1

    metadata = json.loads(metadata_path.read_text())
    model_key = metadata.get("generation_config", {}).get("model_key", "unknown")
    problem_ids = metadata.get("problem_ids", [])

    if not problem_ids:
        console.print("[red]Error: no problem_ids in batch_metadata.json[/red]")
        return 1

    # Find workspaces and load metrics
    all_metrics: list[ProblemMetrics] = []
    missing: list[str] = []

    for pid in sorted(problem_ids):
        ws = find_existing_workspace(pid, model_key, args.workspace_base)
        if not ws:
            missing.append(pid)
            continue
        pm = load_problem_metrics(pid, ws)
        if pm:
            all_metrics.append(pm)
        else:
            missing.append(pid)

    if missing:
        console.print(
            f"[yellow]Warning: missing metrics for {len(missing)} problems: "
            f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}[/yellow]"
        )

    if not all_metrics:
        console.print("[red]Error: no metrics found[/red]")
        return 1

    console.print(f"Loaded metrics for {len(all_metrics)}/{len(problem_ids)} problems")

    if args.json:
        output = build_json_output(
            all_metrics, metadata.get("model", model_key), results_dir
        )
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print_summary(all_metrics, metadata.get("model", model_key), results_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
