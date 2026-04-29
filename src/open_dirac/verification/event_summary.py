"""Summarize EVENT_LOG.jsonl into the text blocks the diagnosis prompt consumes."""

from __future__ import annotations

import json

# Scaffold events surfaced in the diagnosis timeline — everything else is filtered out.
KEY_EVENT_TYPES = frozenset(
    {
        "api_retry",
        "forced_final_call",
        "progress_check",
        "tool_call_failure_fallback",
        "p1_budget_override",
        "p2_stale_loop_override",
        "p3_forced_critic",
        "p4_refuted_recompute",
        "p5_stall_block",
        "compute_verdict_failed",
        "compute_verdict_stall_escalation",
        "termination_blocked",
        "dispatch_failure",
    }
)


def _summarize_llm_calls(llm_calls: list[dict]) -> str | None:
    """Per-agent table: call count, token totals, average duration."""
    if not llm_calls:
        return None
    agent_stats: dict[str, dict] = {}
    for c in llm_calls:
        agent = c.get("agent", "unknown")
        s = agent_stats.setdefault(agent, {"count": 0, "in": 0, "out": 0, "dur": 0.0})
        s["count"] += 1
        s["in"] += c.get("input_tokens", 0)
        s["out"] += c.get("output_tokens", 0)
        s["dur"] += c.get("duration_s", 0.0)

    lines = ["### LLM Calls by Agent", ""]
    lines.append("| Agent | Calls | Input Tok | Output Tok | Avg Duration |")
    lines.append("|-------|------:|----------:|-----------:|-------------:|")
    for agent in sorted(agent_stats):
        s = agent_stats[agent]
        avg = s["dur"] / s["count"] if s["count"] else 0
        lines.append(
            f"| {agent} | {s['count']} | {s['in']:,} | {s['out']:,} | {avg:.1f}s |"
        )
    total_in = sum(s["in"] for s in agent_stats.values())
    total_out = sum(s["out"] for s in agent_stats.values())
    lines.append(f"| **Total** | {len(llm_calls)} | {total_in:,} | {total_out:,} | |")
    return "\n".join(lines)


def _summarize_scaffold_events(scaffold_events: list[dict]) -> str | None:
    """Event counts grouped by category."""
    if not scaffold_events:
        return None
    cat_counts: dict[str, dict[str, int]] = {}
    for e in scaffold_events:
        cat = e.get("category", "unknown")
        event = e.get("event", "unknown")
        cat_counts.setdefault(cat, {})
        cat_counts[cat][event] = cat_counts[cat].get(event, 0) + 1

    lines = ["### Scaffold Events by Category", ""]
    for cat in sorted(cat_counts):
        events_str = ", ".join(
            f"{ev}({n})" for ev, n in sorted(cat_counts[cat].items())
        )
        lines.append(f"- **{cat}:** {events_str}")
    return "\n".join(lines)


def _summarize_key_event_timeline(
    scaffold_events: list[dict], limit: int = 30
) -> str | None:
    """Timeline of overrides, stalls, bailouts, retries, verdict failures."""
    key_events = [e for e in scaffold_events if e.get("event", "") in KEY_EVENT_TYPES]
    if not key_events:
        return None
    lines = ["### Key Event Timeline", ""]
    for e in key_events[:limit]:
        lines.append(
            f"- iter {e.get('iter', '?')}: **{e.get('event', '')}** — {e.get('detail', '')}"
        )
    if len(key_events) > limit:
        lines.append(f"- ... ({len(key_events) - limit} more)")
    return "\n".join(lines)


def summarize_event_log(raw_text: str, max_chars: int = 4096) -> str:
    """Parse EVENT_LOG.jsonl lines into a structured text summary.

    Produces up to three sections: LLM call table, scaffold-event categories,
    and a key-event timeline. Truncated to ``max_chars`` if longer.
    """
    if not raw_text.strip():
        return ""

    llm_calls: list[dict] = []
    scaffold_events: list[dict] = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        kind = entry.get("kind", "")
        if kind == "llm_call":
            llm_calls.append(entry)
        elif kind == "scaffold":
            scaffold_events.append(entry)

    sections = [
        s
        for s in (
            _summarize_llm_calls(llm_calls),
            _summarize_scaffold_events(scaffold_events),
            _summarize_key_event_timeline(scaffold_events),
        )
        if s
    ]
    if not sections:
        return ""

    result = "\n\n".join(sections)
    if len(result) > max_chars:
        result = result[: max_chars - 20] + "\n\n[... truncated]"
    return result
