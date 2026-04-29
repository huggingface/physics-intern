"""Shared helpers for OpenAI-compatible providers (HuggingFace, vLLM).

Internal module — not part of the public provider API.

These helpers factor out byte-identical logic from ``huggingface.py`` and
``vllm.py``. The streaming accumulator is intentionally NOT shared — the two
providers have subtle divergences (reasoning-delta dispatch, JSON-failure
fallback) that make merging risky; see ``cleaning_plan.md`` Slice 1B.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)


def canonical_tool_arguments_json_string(arguments: str | None) -> str:
    """Return JSON text valid for ``tool_calls[].function.arguments`` on the wire.

    OpenAI-compatible servers (including vLLM) typically ``json.loads`` this
    inner string when handling the next turn. Streaming accumulation can
    produce **invalid JSON** (truncation, bad escapes); :mod:`vllm` then falls
    back to ``{"raw": ...}`` for *local* tool execution but must not forward
    the broken string in :meth:`format_assistant_message`, or the **following**
    request fails with ``HTTP 400`` (e.g. ``Expecting ',' delimiter``,
    ``Invalid \\escape``).

    If *arguments* is valid JSON, it is re-serialized compactly. If not, a
    small diagnostic object is emitted so the outbound request always carries
    parseable tool JSON.
    """
    if arguments is None or not str(arguments).strip():
        return "{}"
    s = str(arguments).strip()
    try:
        parsed = json.loads(s)
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError as exc:
        snippet = s[:4000]
        logger.warning(
            "Invalid tool-call arguments JSON from model; substituting "
            "diagnostic payload for API resend (%s at char %s): %r...",
            exc.msg,
            exc.pos if exc.pos is not None else "?",
            snippet[:120],
        )
        diagnostic: dict[str, bool | str | int | None] = {
            "_open_dirac_tool_arguments_parse_error": True,
            "parse_error_message": exc.msg,
            "parse_error_pos": exc.pos,
            "snippet": snippet,
        }
        return json.dumps(diagnostic, ensure_ascii=False, separators=(",", ":"))


def strip_tool_messages(messages: list[dict]) -> list[dict]:
    """Remove tool-call artifacts from messages for text-only calls.

    OSS models served via OpenAI-compatible endpoints may hallucinate tool
    calls when the conversation history contains tool-call messages, even
    when no tools are offered on the current turn.  Stripping these prevents
    ``output_parse_failed`` and "Tool choice is none, but model called a
    tool" errors.
    """
    cleaned = []
    for msg in messages:
        if msg.get("role") == "tool":
            continue
        if "tool_calls" in msg:
            msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            if not msg.get("content"):
                msg["content"] = "[prior tool interaction omitted]"
        cleaned.append(msg)
    return cleaned


def build_raw_tool_call(tc_id: str, name: str, arguments: str) -> SimpleNamespace:
    """Build one raw_content tool-call entry with the OpenAI SDK shape.

    ``format_assistant_message`` expects ``raw_content.tool_calls`` to be
    iterable of objects with ``.id`` and ``.function.name`` / ``.function.arguments``.
    """
    return SimpleNamespace(
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
