"""Markdown parsing with YAML frontmatter support."""

import re
import yaml


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SECTION_RE = re.compile(r"^## .+", re.MULTILINE)

# Shared critique regex constants
CRIT_ID_RE = re.compile(r"CRIT(?:IQUE)?-\d+")
CRIT_HEADER_RE = re.compile(r"^#{2,3} CRIT(?:IQUE)?-\d+", re.MULTILINE)
CRIT_UNRESOLVED_RE = re.compile(
    r"#{2,3} CRIT(?:IQUE)?-\d+\s*\[(\w+)\]\s*\[UNRESOLVED\]"
)
CRIT_WITHDRAWN_RE = re.compile(r"#{2,3} CRIT(?:IQUE)?-\d+\s*\[(\w+)\]\s*\[WITHDRAWN\]")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split text into (frontmatter_dict, body). Returns ({}, text) on failure."""
    # Strip code fences wrapping frontmatter (LLMs sometimes emit ```yaml\n---\n...\n---\n```)
    stripped = re.sub(r"^```\w*\s*\n", "", text)
    stripped = re.sub(r"\n```\s*(?:\n|$)", "\n", stripped, count=1)
    match = FRONTMATTER_RE.match(stripped)
    if match:
        text = stripped
    else:
        match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    yaml_str = match.group(1)
    body = text[match.end() :]
    try:
        meta = yaml.safe_load(yaml_str)
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError:
        meta = _fallback_parse(yaml_str)
    return meta, body


def _fallback_parse(yaml_str: str) -> dict:
    """Regex fallback for broken YAML frontmatter."""
    result = {}
    for line in yaml_str.strip().splitlines():
        m = re.match(r"^(\w[\w_]*)\s*:\s*(.+)$", line)
        if m:
            key = m.group(1)
            val = m.group(2).strip().strip('"').strip("'")
            # Try to parse as int
            try:
                val = int(val)
            except (ValueError, TypeError):
                pass
            result[key] = val
    return result


def render_frontmatter(meta: dict, body: str) -> str:
    """Render a dict + body back into frontmatter Markdown."""
    yaml_str = yaml.dump(meta, default_flow_style=False, sort_keys=False).strip()
    return f"---\n{yaml_str}\n---\n\n{body}"


def tail_entries(text: str, n: int) -> str:
    """Return the last N '## ' sections from text."""
    parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    # First part is everything before the first ## section (frontmatter + intro)
    sections = [p for p in parts if p.startswith("## ")]
    if not sections:
        return text
    selected = sections[-n:]
    return "\n".join(selected)


def extract_section_by_id(text: str, section_id: str) -> str:
    """Extract a section whose heading contains section_id (e.g. 'CRIT-010')."""
    parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    for part in parts:
        if section_id in part:
            return part.strip()
    return ""


def count_unresolved_critiques(text: str) -> int:
    """Count unresolved critiques from CRITIQUE_LOG.md content."""
    return len(CRIT_UNRESOLVED_RE.findall(text))


def insert_into_active_critiques(text: str, new_content: str) -> str:
    """Insert new critique entries under '# Active Critiques', before '# Resolved Critiques'.

    If the headings are missing, falls back to appending to the body.
    """
    active_marker = "# Active Critiques"
    resolved_marker = "# Resolved Critiques"

    active_idx = text.find(active_marker)
    resolved_idx = text.find(resolved_marker)

    if active_idx == -1:
        # No Active section — append to end of body
        return text + "\n" + new_content

    if resolved_idx == -1 or resolved_idx < active_idx:
        # No Resolved section after Active — append after Active content
        return text + "\n" + new_content

    # Insert new content just before the "# Resolved Critiques" heading
    return (
        text[:resolved_idx].rstrip()
        + "\n\n"
        + new_content.strip()
        + "\n\n"
        + text[resolved_idx:]
    )


_RETRACTION_PATTERNS = [
    re.compile(
        r"reproduction succeeded.*no (issues? found|flaws?|errors?)", re.IGNORECASE
    ),
    re.compile(r"no issues? found.*none needed", re.IGNORECASE),
    re.compile(r"documenting successful verification", re.IGNORECASE),
    re.compile(r"no flaw(s)? (found|identified|detected)", re.IGNORECASE),
    re.compile(r"does not warrant filing", re.IGNORECASE),
    re.compile(r"not filing", re.IGNORECASE),
    re.compile(
        r"no (genuine )?(objection|critique|issue)s? to (file|raise|report)",
        re.IGNORECASE,
    ),
]

_CRITIQUE_HEADER_RE = CRIT_HEADER_RE


def filter_self_retracted_critiques(response_text: str) -> tuple[str, list[str]]:
    """Filter out self-retracted LOW critiques from a critic response.

    A critique is self-retracted if it is LOW severity AND Phase 2 contains
    retraction signals (e.g. "reproduction succeeded, no issues found").

    Also handles the NO_CRITIQUES_FILED marker line.

    Returns (filtered_text, retracted_summaries) where summaries are
    one-liners like "CRIT-012 targeting WH-003: self-retracted".
    """
    # Handle NO_CRITIQUES_FILED marker
    if re.search(r"^NO_CRITIQUES_FILED:", response_text, re.MULTILINE):
        return ("", [])

    # Split into individual critique blocks
    splits = _CRITIQUE_HEADER_RE.split(response_text)
    headers = _CRITIQUE_HEADER_RE.findall(response_text)

    if not headers:
        return (response_text, [])

    preamble = splits[0]  # text before the first critique header
    blocks = list(zip(headers, splits[1:]))

    kept: list[str] = []
    retracted: list[str] = []

    for header, body in blocks:
        full_block = header + body
        # Extract severity from the header line (first line of body before newline)
        first_line = (header + body.split("\n")[0]).upper()
        is_retractable = "[LOW]" in first_line or "[MEDIUM]" in first_line

        if is_retractable and _is_self_retracted(body):
            # Build summary
            crit_id = CRIT_ID_RE.search(header).group()
            target_match = re.search(r"\*\*Target:\*\*\s*(\S+)", body)
            target = target_match.group(1) if target_match else "unknown"
            retracted.append(f"{crit_id} targeting {target}: self-retracted")
            # Mark as WITHDRAWN instead of removing
            withdrawn_block = full_block.replace("[UNRESOLVED]", "[WITHDRAWN]")
            kept.append(withdrawn_block)
        else:
            kept.append(full_block)

    filtered_text = preamble.rstrip()
    if kept:
        if filtered_text:
            filtered_text += "\n\n"
        filtered_text += "\n".join(kept)

    return (filtered_text.strip(), retracted)


def _is_self_retracted(block_body: str) -> bool:
    """Check if a critique block's Phase 2 contains retraction signals."""
    # Extract text after Phase 2 heading
    phase2_match = re.search(r"###\s*Phase\s*2", block_body, re.IGNORECASE)
    if not phase2_match:
        # No Phase 2 section — check the whole body
        text_to_check = block_body
    else:
        text_to_check = block_body[phase2_match.start() :]

    return any(pat.search(text_to_check) for pat in _RETRACTION_PATTERNS)


def resolve_critique(text: str, crit_id: str, resolution_note: str) -> str:
    """Move a critique from Active to Resolved, changing [UNRESOLVED] -> [RESOLVED].

    Adds a resolution note. Returns text unchanged if crit_id is not found
    or already resolved.

    Captures the full block (including ### Phase 1 / Phase 2 sub-headings)
    by stopping only at headings of the same or higher level.
    """
    # Find the critique header to determine its heading level
    header_re = re.compile(
        r"^(#{2,3}) CRIT(?:IQUE)?-"
        + re.escape(crit_id.split("-")[-1])
        + r"\s*\[\w+\]\s*\[UNRESOLVED\]",
        re.MULTILINE,
    )
    header_match = header_re.search(text)
    if not header_match:
        return text  # Not found or already resolved

    heading_level = len(header_match.group(1))  # 2 or 3
    start = header_match.start()

    # Capture until next heading of same or higher level (fewer #), or end-of-string
    block_end_re = re.compile(r"\n#{1," + str(heading_level) + r"} ")
    end_match = block_end_re.search(text, header_match.end())
    end = end_match.start() if end_match else len(text)

    block = text[start:end]

    # Mark as resolved and add resolution note
    resolved_block = block.replace("[UNRESOLVED]", "[RESOLVED]")
    resolved_block = (
        resolved_block.rstrip() + f"\n- **Resolution:** {resolution_note}\n"
    )

    # Remove from current position
    text_without = text[:start] + text[end:]

    # Append to Resolved Critiques section
    resolved_marker = "# Resolved Critiques"
    resolved_idx = text_without.find(resolved_marker)
    if resolved_idx == -1:
        # No Resolved section — append at end
        return text_without.rstrip() + "\n\n# Resolved Critiques\n\n" + resolved_block
    else:
        insert_pos = resolved_idx + len(resolved_marker)
        return (
            text_without[:insert_pos].rstrip()
            + "\n\n"
            + resolved_block.strip()
            + "\n"
            + text_without[insert_pos:]
        )


def count_withdrawn_critiques(text: str) -> int:
    """Count critiques marked as WITHDRAWN."""
    return len(CRIT_WITHDRAWN_RE.findall(text))


def recount_critique_metadata(content: str) -> dict:
    """Recount unresolved and total critiques from CRITIQUE_LOG content.

    Returns dict with keys: unresolved_critiques, total_critiques,
    withdrawn_critiques.
    """
    unresolved = count_unresolved_critiques(content)
    total = len(CRIT_HEADER_RE.findall(content))
    withdrawn = count_withdrawn_critiques(content)
    return {
        "unresolved_critiques": unresolved,
        "total_critiques": total,
        "withdrawn_critiques": withdrawn,
    }


def ensure_critique_metadata_consistent(content: str) -> str:
    """Recount critique statistics and write them into frontmatter atomically.

    This is the single source of truth for critique metadata. Both the
    orchestrator (after resolving critiques) and the critic (after filing
    critiques) call this to keep counts accurate.
    """
    meta, body = parse_frontmatter(content)
    recounted = recount_critique_metadata(content)
    meta.update(recounted)
    return render_frontmatter(meta, body)


# --- Nested bracket flattening ---

_NESTED_UNVERIFIED_RE = re.compile(
    r"\[+\s*((?:COMP|TASK)-\d+)(?::unverified)?(?:\]:unverified)*\s*\]"
)


def flatten_unverified_brackets(text: str) -> str:
    """Collapse nested [[[ID:unverified]:unverified]...] to [ID:unverified]."""
    return _NESTED_UNVERIFIED_RE.sub(r"[\1:unverified]", text)


# --- Computation log parsing and stall detection ---

_COMP_HEADER_RE = re.compile(r"^## (?:COMP|TASK)-\d+", re.MULTILINE)
_ER_WH_ID_RE = re.compile(r"(?:ER|WH)-\d+")

# ER/WH section detection — matches both ## headers and **bold** line-start formats.
# LLMs sometimes write **ER-001 — Title** instead of ## ER-001 — Title.
_ER_SECTION_RE = re.compile(r"^(?:#{2,3} |\*\*)(ER-\d+)", re.MULTILINE)
_WH_SECTION_RE = re.compile(r"^(?:#{2,3} |\*\*)(WH-\d+)", re.MULTILINE)


def count_er_sections(text: str) -> int:
    """Count ER-NNN section entries (H2/H3 headers or bold-line-start)."""
    return len(_ER_SECTION_RE.findall(text))


def count_wh_sections(text: str) -> int:
    """Count WH-NNN section entries (H2/H3 headers or bold-line-start)."""
    return len(_WH_SECTION_RE.findall(text))


def find_er_section_ids(text: str) -> list[str]:
    """Extract ER-NNN IDs from section entries (H2/H3 headers or bold-line-start)."""
    return _ER_SECTION_RE.findall(text)


def normalize_er_wh_headers(text: str) -> str:
    r"""Convert \*\*ER-NNN ...\*\* and \*\*WH-NNN ...\*\* bold lines to ## headers."""
    return re.sub(
        r"^\*\*((?:ER|WH)-\d+[^*]*?)\*\*\s*$",
        r"## \1",
        text,
        flags=re.MULTILINE,
    )


def _parse_comp_entries(text: str) -> list[dict]:
    """Parse EVIDENCE_LOG.md into structured entries.

    Returns list of {"id": "COMP-001", "claim": "...", "verdict": "VERIFIED"|..., "result": "..."}.
    """
    splits = _COMP_HEADER_RE.split(text)
    headers = _COMP_HEADER_RE.findall(text)
    if not headers:
        return []

    entries = []
    for header, body in zip(headers, splits[1:]):
        entry_id_match = re.search(r"(?:COMP|TASK)-\d+", header)
        entry_id = entry_id_match.group() if entry_id_match else "UNKNOWN"

        # Extract CLAIM line — handle colon inside or outside bold markers
        claim_match = re.search(
            r"\*\*(?:CLAIM|Task|Claim):?\*{0,2}:?\s*(.+)", body, re.IGNORECASE
        )
        claim = claim_match.group(1).strip() if claim_match else ""

        # Extract VERDICT line — handle **VERDICT:** or **VERDICT**: or **VERDICT**
        verdict_match = re.search(
            r"\*\*VERDICT:?\*{0,2}:?\s*(\w+)", body, re.IGNORECASE
        )
        verdict = verdict_match.group(1).strip().upper() if verdict_match else ""

        # Extract RESULT block: text between **RESULT**: and next ** header or ## header
        result_match = re.search(
            r"\*\*(?:RESULT|Result)\*?\*?:?\s*\n(.*?)(?=\n\*\*[A-Z]|\n## |\Z)",
            body,
            re.DOTALL | re.IGNORECASE,
        )
        result = result_match.group(1).strip() if result_match else ""

        # Extract METHOD block
        method_match = re.search(
            r"\*\*(?:METHOD|Method)\*?\*?:?\s*\n(.*?)(?=\n\*\*[A-Z]|\n## |\Z)",
            body,
            re.DOTALL | re.IGNORECASE,
        )
        method = method_match.group(1).strip() if method_match else ""

        # Extract NOTES block
        notes_match = re.search(
            r"\*\*(?:NOTES?|Notes?)\*?\*?:?\s*\n(.*?)(?=\n\*\*[A-Z]|\n## |\Z)",
            body,
            re.DOTALL | re.IGNORECASE,
        )
        notes = notes_match.group(1).strip() if notes_match else ""

        entries.append(
            {
                "id": entry_id,
                "claim": claim,
                "verdict": verdict,
                "result": result,
                "method": method,
                "notes": notes,
                "body": body,
            }
        )
    return entries


def detect_computation_stalls(text: str, threshold: int = 3) -> list[dict]:
    """Find claims with >= threshold consecutive non-VERIFIED verdicts.

    Returns [{"claim": str, "count": int, "verdicts": list[str]}].
    Claim matching: extract ER/WH IDs via regex; fall back to first-80-char prefix.
    A VERIFIED verdict resets the consecutive count.
    """
    entries = _parse_comp_entries(text)
    # Group by normalized claim key
    claim_streaks: dict[
        str, list[str]
    ] = {}  # key -> current streak of non-VERIFIED verdicts

    for entry in entries:
        key = _normalize_claim_key(entry["claim"])
        if not key:
            continue
        if entry["verdict"] == "VERIFIED":
            claim_streaks[key] = []  # reset
        else:
            if key not in claim_streaks:
                claim_streaks[key] = []
            claim_streaks[key].append(entry["verdict"] or "UNKNOWN")

    stalls = []
    for key, verdicts in claim_streaks.items():
        if len(verdicts) >= threshold:
            stalls.append(
                {
                    "claim": key,
                    "count": len(verdicts),
                    "verdicts": verdicts,
                }
            )
    return stalls


def _normalize_claim_key(claim: str) -> str:
    """Normalize a claim to a matching key.

    Prefers ER/WH IDs if present; falls back to first-80-char prefix (lowered, stripped).
    """
    ids = _ER_WH_ID_RE.findall(claim)
    if ids:
        return " ".join(sorted(set(ids)))
    # Fallback: first 80 chars, lowercased, whitespace-collapsed
    normalized = " ".join(claim.lower().split())[:80]
    return normalized


def detect_zero_output_stalls(text: str) -> list[dict]:
    """Find computation entries where the agent produced no text output.

    Returns [{"claim": normalized_key, "entry_id": entry_id}].
    """
    entries = _parse_comp_entries(text)
    stalls = []
    for entry in entries:
        if "Agent produced no text output" in entry.get("body", ""):
            key = _normalize_claim_key(entry["claim"])
            if key:
                stalls.append({"claim": key, "entry_id": entry["id"]})
    return stalls


def _format_failure_excerpt(entry: dict) -> str:
    """Combine METHOD + RESULT + NOTES + VERDICT into a formatted failure excerpt."""
    parts = []
    if entry.get("verdict"):
        parts.append(f"**Verdict:** {entry['verdict']}")
    if entry.get("method"):
        parts.append(f"**Method:** {entry['method']}")
    if entry.get("result"):
        parts.append(f"**Result:** {entry['result']}")
    if entry.get("notes"):
        parts.append(f"**Notes:** {entry['notes']}")
    return "\n".join(parts)


def find_prior_failures_for_claim(comp_log: str, task_body: str) -> list[str]:
    """Find failure excerpts from prior non-VERIFIED computations matching the claim in task_body.

    Matching: extract ER/WH IDs from task_body; match against claim lines in COMP entries.
    Falls back to first-80-char normalized prefix matching.
    Returns formatted excerpts (METHOD + RESULT + NOTES + VERDICT), most recent first.
    """
    entries = _parse_comp_entries(comp_log)
    task_key = _normalize_claim_key(task_body)
    if not task_key:
        return []

    # Also extract raw IDs from task_body for ID-based matching
    task_ids = set(_ER_WH_ID_RE.findall(task_body))

    results = []
    for entry in entries:
        if entry["verdict"] == "VERIFIED":
            continue
        entry_key = _normalize_claim_key(entry["claim"])
        entry_ids = set(_ER_WH_ID_RE.findall(entry["claim"]))

        matched = False
        # Match by ER/WH IDs (if both have them and they overlap)
        if task_ids and entry_ids and task_ids & entry_ids:
            matched = True
        elif not task_ids and not entry_ids and entry_key == task_key:
            # Fallback: prefix matching (only when neither has IDs)
            matched = True

        if matched:
            excerpt = _format_failure_excerpt(entry)
            if excerpt:
                results.append(excerpt)

    # Most recent first (entries are in document order, so reverse)
    results.reverse()
    return results
