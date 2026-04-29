"""Researcher agent: one-shot analytical reasoning with structured JSON output."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from open_dirac.llm import LLMResponse, ParseFailureError
from open_dirac.research_state import Evidence

from ..evidence_base import ENTITY_ID_RE, EvidenceAgent
from ..parsing import JSON_FENCE_RE, try_json_loads

if TYPE_CHECKING:
    from open_dirac.task import Task

# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_VALID_CONFIDENCE = {"exact", "approximate", "partial"}

# Regex to find the start of the last ```json fence (used to split derivation from JSON)
_LAST_JSON_FENCE_RE = re.compile(r"```json\s*\n", re.DOTALL)


def _extract_derivation_text(text: str) -> str:
    """Extract derivation text: everything before the last ```json fence.

    If no fence is found, returns the full text (useful on parse failure).
    The JSON block itself is excluded from the derivation file.
    """
    matches = list(_LAST_JSON_FENCE_RE.finditer(text))
    if matches:
        return text[: matches[-1].start()].rstrip()
    return text


def _parse_researcher_json(text: str) -> dict | None:
    """Extract the last JSON block containing a 'result' key from model output.

    Tries fenced ```json blocks first (last match), then falls back to
    brace-counting for bare JSON (needed because the reasoning field may
    contain braces).
    """
    # Prefer fenced ```json blocks — take the last one
    fenced = list(JSON_FENCE_RE.finditer(text))
    if fenced:
        try:
            parsed = try_json_loads(fenced[-1].group(1).strip())
            if isinstance(parsed, dict) and "result" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Bare JSON fallback: brace-counting (like critic)
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                if '"result"' in candidate:
                    try:
                        parsed = try_json_loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
                start = None

    return None


class ResearcherAgent(EvidenceAgent):
    name = "researcher"
    prompt_file = "prompt.md"
    tools = []  # one-shot: no tools
    raise_on_parse_failure = True

    def _validate_response(self, response: LLMResponse) -> bool:
        return _parse_researcher_json(response.text or "") is not None

    def _parse_retry_hint(self, parse_error: str | None = None) -> str:
        return (
            "Recall the required output format and provide it now:\n\n"
            "```json\n"
            "{\n"
            '  "result": "Compact conclusion (quotable in one paragraph)",\n'
            '  "method": "Analytical approach name",\n'
            '  "confidence": "exact|approximate|partial",\n'
            '  "summary": "One-sentence summary"\n'
            "}\n"
            "```"
        )

    def process_response(self, response: LLMResponse, task: Task, iteration: int):
        """Parse structured JSON from one-shot response text and build Evidence."""
        text = response.text or ""
        parsed = _parse_researcher_json(text)

        # Resolve target ID early (needed for derivation filename)
        target_id = task.target_claim or ""
        if not target_id and self.research_state:
            ids = ENTITY_ID_RE.findall(task.body or "")
            target_id = ids[0] if ids else ""

        # Save derivation file (everything before the last ```json fence)
        derivation_file = ""
        workspace = getattr(self, "workspace", None)
        if text and workspace:
            derivation_text = _extract_derivation_text(text)
            if derivation_text.strip():
                safe_target = target_id or "unknown"
                derivation_file = f"{safe_target}_{iteration:03d}.md"
                workspace.write_file(f"derivations/{derivation_file}", derivation_text)

        if parsed and "result" in parsed:
            confidence = parsed.get("confidence", "partial")
            if confidence not in _VALID_CONFIDENCE:
                confidence = "partial"
            evidence = Evidence(
                type="research",
                reasoning=text,  # full derivation text (backward compat)
                result=parsed.get("result", ""),
                method=parsed.get("method", ""),
                confidence=confidence,
                summary=parsed.get("summary", ""),
                iteration=iteration,
                derivation_file=derivation_file,
            )
        else:
            # Unreachable when raise_on_parse_failure=True — _call_with_retry
            # raises ParseFailureError before process_response is called.
            raise ParseFailureError(
                agent_name=self.name,
                detail="process_response reached without valid parsed output",
            )

        # Store on target entity — use task.target_claim, not tool params
        if self.research_state:
            self._store_evidence(target_id, evidence)
