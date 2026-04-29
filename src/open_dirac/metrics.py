"""Metrics tracking for OpenDirac iterations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class CallRecord:
    iteration: int
    agent: str
    input_tokens: int
    output_tokens: int
    duration: float
    max_tokens_hit: bool
    rounds: int = 1
    tool_calls: int = 0
    truncated: bool = False
    reasoning_tokens: int = 0
    answer_tokens: int = 0


# ---------------------------------------------------------------------------
# METRICS.md parsers (shared with scripts/analyze_batch.py)
# ---------------------------------------------------------------------------


def parse_yaml_frontmatter(text: str) -> dict:
    """Extract YAML frontmatter from a markdown file.

    Returns {} on missing or malformed frontmatter — never raises.
    """
    m = re.match(r"^---\n(.*?\n)---", text, re.DOTALL)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}


def parse_metrics_table(text: str) -> list[dict]:
    """Parse the per-iteration markdown table into rows.

    Handles both table variants emitted by ``MetricsTracker.to_markdown``:
    - 8 columns: iter | agent | in | out | max_hit | rounds | tool_calls | duration
    - 6 columns: iter | agent | in | out | max_hit | duration   (no tool use)

    Unknown / malformed rows are skipped silently.
    """
    rows: list[dict] = []
    in_table = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("| Iter"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            parts = [c.strip() for c in line.split("|")[1:-1]]
            try:
                if len(parts) >= 8:
                    rows.append(
                        {
                            "iter": int(parts[0]),
                            "agent": parts[1],
                            "input_tokens": int(parts[2]),
                            "output_tokens": int(parts[3]),
                            "max_tokens_hit": parts[4].lower() == "yes",
                            "rounds": int(parts[5]),
                            "tool_calls": int(parts[6]),
                            "duration_s": float(parts[7]),
                        }
                    )
                elif len(parts) >= 6:
                    rows.append(
                        {
                            "iter": int(parts[0]),
                            "agent": parts[1],
                            "input_tokens": int(parts[2]),
                            "output_tokens": int(parts[3]),
                            "max_tokens_hit": parts[4].lower() == "yes",
                            "rounds": 1,
                            "tool_calls": 0,
                            "duration_s": float(parts[5]),
                        }
                    )
            except (ValueError, IndexError):
                pass
        elif in_table and not line.startswith("|"):
            break
    return rows


def _parse_alerts(text: str) -> list[dict]:
    """Parse the ``# Alerts`` section into a list of {iteration, message} dicts."""
    alerts: list[dict] = []
    in_section = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("# Alerts"):
            in_section = True
            continue
        if in_section:
            if line.startswith("#"):
                break
            m = re.match(r"^-\s*\[iter\s+(\d+)\]\s*(.*)$", line)
            if m:
                try:
                    alerts.append(
                        {"iteration": int(m.group(1)), "message": m.group(2).strip()}
                    )
                except ValueError:
                    pass
    return alerts


class MetricsTracker:
    """Track per-iteration metrics, alerts, and file sizes."""

    def __init__(self):
        self.calls: list[CallRecord] = []
        self.alerts: list[dict] = []
        self.last_critic_iteration: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_reasoning_tokens: int = 0
        self.total_answer_tokens: int = 0
        self.max_tokens_reached_count: int = 0
        self.total_tool_calls: int = 0

    def record_call(
        self,
        iteration: int,
        agent: str,
        input_tokens: int,
        output_tokens: int,
        duration: float,
        max_tokens_hit: bool,
        rounds: int = 1,
        tool_calls: int = 0,
        truncated: bool = False,
        reasoning_tokens: int = 0,
        answer_tokens: int = 0,
    ):
        self.calls.append(
            CallRecord(
                iteration=iteration,
                agent=agent,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration=duration,
                max_tokens_hit=max_tokens_hit,
                rounds=rounds,
                tool_calls=tool_calls,
                truncated=truncated,
                reasoning_tokens=reasoning_tokens,
                answer_tokens=answer_tokens,
            )
        )
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_reasoning_tokens += reasoning_tokens
        self.total_answer_tokens += answer_tokens
        self.total_tool_calls += tool_calls
        if max_tokens_hit:
            self.max_tokens_reached_count += 1
        if agent == "deep_critic":
            self.last_critic_iteration = iteration

    def alert(self, iteration: int, message: str):
        self.alerts.append({"iteration": iteration, "message": message})

    @classmethod
    def load(cls, workspace_path: Path) -> "MetricsTracker":
        """Rehydrate a tracker from a workspace's ``METRICS.md``.

        Aggregate counters (``total_input_tokens``, etc.) are seeded from the
        YAML frontmatter, which is authoritative — in particular because
        ``reasoning_tokens`` and ``answer_tokens`` are not present per-row in
        the rendered table. Per-iteration ``CallRecord``s are reconstructed
        from the table for agent-breakdown reporting; fields absent from the
        table default to 0 / False.

        Missing or malformed ``METRICS.md`` → returns an empty tracker. Per the
        project invariant, parse failures never raise.
        """
        tracker = cls()
        metrics_path = Path(workspace_path) / "METRICS.md"
        if not metrics_path.exists():
            return tracker
        try:
            text = metrics_path.read_text()
        except OSError:
            return tracker
        if not text.strip():
            return tracker

        fm = parse_yaml_frontmatter(text)
        if fm:
            tracker.total_input_tokens = int(fm.get("total_input_tokens", 0) or 0)
            tracker.total_output_tokens = int(fm.get("total_output_tokens", 0) or 0)
            tracker.total_reasoning_tokens = int(
                fm.get("total_reasoning_tokens", 0) or 0
            )
            tracker.total_answer_tokens = int(fm.get("total_answer_tokens", 0) or 0)
            tracker.total_tool_calls = int(fm.get("total_tool_calls", 0) or 0)
            tracker.max_tokens_reached_count = int(
                fm.get("max_tokens_reached_count", 0) or 0
            )

        for row in parse_metrics_table(text):
            tracker.calls.append(
                CallRecord(
                    iteration=row["iter"],
                    agent=row["agent"],
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    duration=row["duration_s"],
                    max_tokens_hit=row["max_tokens_hit"],
                    rounds=row["rounds"],
                    tool_calls=row["tool_calls"],
                    truncated=False,
                    reasoning_tokens=0,
                    answer_tokens=0,
                )
            )
            if row["agent"] == "deep_critic":
                tracker.last_critic_iteration = max(
                    tracker.last_critic_iteration, row["iter"]
                )

        tracker.alerts = _parse_alerts(text)
        return tracker

    def to_markdown(self) -> str:
        """Render metrics as METRICS.md content."""
        total_iters = max((c.iteration for c in self.calls), default=0)
        has_tool_calls = any(c.tool_calls > 0 for c in self.calls)

        meta = {
            "total_iterations": total_iters,
            "total_llm_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "max_tokens_reached_count": self.max_tokens_reached_count,
        }
        if self.total_reasoning_tokens > 0:
            meta["total_reasoning_tokens"] = self.total_reasoning_tokens
            meta["total_answer_tokens"] = self.total_answer_tokens
        if has_tool_calls:
            meta["total_tool_calls"] = self.total_tool_calls

        lines = ["---"]
        for k, v in meta.items():
            lines.append(f"{k}: {v}")
        lines.append("---\n")

        lines.append("# Per-Iteration Metrics\n")
        if has_tool_calls:
            lines.append(
                "| Iter | Agent | Input Tokens | Output Tokens | Max Tokens Hit | Rounds | Tool Calls | Duration (s) |"
            )
            lines.append(
                "|------|-------|-------------|---------------|----------------|--------|------------|-------------|"
            )
            for c in reversed(self.calls):
                lines.append(
                    f"| {c.iteration} | {c.agent} | {c.input_tokens} | {c.output_tokens} "
                    f"| {'yes' if c.max_tokens_hit else 'no'} | {c.rounds} | {c.tool_calls} | {c.duration:.1f} |"
                )
        else:
            lines.append(
                "| Iter | Agent | Input Tokens | Output Tokens | Max Tokens Hit | Duration (s) |"
            )
            lines.append(
                "|------|-------|-------------|---------------|----------------|-------------|"
            )
            for c in reversed(self.calls):
                lines.append(
                    f"| {c.iteration} | {c.agent} | {c.input_tokens} | {c.output_tokens} "
                    f"| {'yes' if c.max_tokens_hit else 'no'} | {c.duration:.1f} |"
                )

        if self.alerts:
            lines.append("\n# Alerts")
            for a in self.alerts:
                lines.append(f"- [iter {a['iteration']}] {a['message']}")

        return "\n".join(lines) + "\n"
