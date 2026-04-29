"""Tests for markdown parsing utilities."""

from pathlib import Path
from open_dirac.utils.markdown import (
    parse_frontmatter,
    render_frontmatter,
    tail_entries,
    extract_section_by_id,
    count_unresolved_critiques,
    insert_into_active_critiques,
    resolve_critique,
    filter_self_retracted_critiques,
    recount_critique_metadata,
    _parse_comp_entries,
    _format_failure_excerpt,
    detect_computation_stalls,
    find_prior_failures_for_claim,
    count_er_sections,
    count_wh_sections,
    find_er_section_ids,
    normalize_er_wh_headers,
    flatten_unverified_brackets,
    detect_zero_output_stalls,
    count_withdrawn_critiques,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_frontmatter_basic():
    text = "---\ntitle: hello\nstatus: ok\n---\n\nBody here."
    meta, body = parse_frontmatter(text)
    assert meta["title"] == "hello"
    assert meta["status"] == "ok"
    assert "Body here." in body


def test_parse_frontmatter_no_frontmatter():
    text = "Just a body with no frontmatter."
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_parse_frontmatter_invalid_yaml():
    text = "---\n: broken yaml [[\n---\n\nBody."
    meta, body = parse_frontmatter(text)
    # Should not crash, returns fallback or empty
    assert isinstance(meta, dict)
    assert "Body." in body


def test_render_frontmatter():
    meta = {"title": "test", "status": "ok"}
    body = "Some content."
    result = render_frontmatter(meta, body)
    assert result.startswith("---\n")
    assert "title: test" in result
    assert "Some content." in result


def test_tail_entries():
    text = (FIXTURES / "computation_log.md").read_text()
    result = tail_entries(text, 1)
    assert "COMP-001" in result
    assert "COMP-002" not in result


def test_tail_entries_all():
    text = (FIXTURES / "computation_log.md").read_text()
    result = tail_entries(text, 5)
    assert "COMP-001" in result
    assert "COMP-002" in result


def test_extract_section_by_id():
    text = (FIXTURES / "critique_log.md").read_text()
    section = extract_section_by_id(text, "CRIT-001")
    assert "HIGH" in section
    assert "UNRESOLVED" in section


def test_extract_section_by_id_not_found():
    text = "## Section A\nContent A\n## Section B\nContent B"
    result = extract_section_by_id(text, "NONEXISTENT")
    assert result == ""


def test_count_unresolved_critiques():
    text = (FIXTURES / "critique_log.md").read_text()
    count = count_unresolved_critiques(text)
    assert count == 2  # 1 HIGH + 1 MEDIUM unresolved


def test_count_unresolved_critiques_critique_prefix():
    """Test that CRITIQUE-NNN format (LLM drift) is also matched."""
    text = """
## CRITIQUE-001 [HIGH] [UNRESOLVED]
- **Target:** WH-1
- **Critique:** Something wrong.

## CRITIQUE-002 [LOW] [UNRESOLVED]
- **Target:** WH-2
"""
    count = count_unresolved_critiques(text)
    assert count == 2  # 1 HIGH + 1 LOW


def test_count_unresolved_critiques_empty():
    count = count_unresolved_critiques("No critiques here.")
    assert count == 0


# --- Tests for insert_into_active_critiques ---


def test_insert_into_active_critiques():
    text = (FIXTURES / "critique_log.md").read_text()
    new_critique = "## CRIT-004 [HIGH] [UNRESOLVED]\n- **Target:** WH-3\n- **Critique:** New issue.\n"
    result = insert_into_active_critiques(text, new_critique)

    # New critique should appear before Resolved section
    active_idx = result.find("# Active Critiques")
    resolved_idx = result.find("# Resolved Critiques")
    crit004_idx = result.find("CRIT-004")
    assert active_idx < crit004_idx < resolved_idx

    # Existing critiques should still be present
    assert "CRIT-001" in result
    assert "CRIT-002" in result
    assert "CRIT-003" in result


def test_insert_into_active_no_sections():
    text = "---\ntotal_critiques: 0\n---\n\nSome body.\n"
    new_critique = "## CRIT-001 [HIGH] [UNRESOLVED]\n- New.\n"
    result = insert_into_active_critiques(text, new_critique)
    assert "CRIT-001" in result


# --- Tests for resolve_critique ---


def test_resolve_critique():
    text = (FIXTURES / "critique_log.md").read_text()
    result = resolve_critique(text, "CRIT-001", "Verified by computation.")

    # CRIT-001 should now be RESOLVED
    assert "[RESOLVED]" in extract_section_by_id(result, "CRIT-001")
    assert "Verified by computation." in result

    # CRIT-001 should be in Resolved section, not Active
    resolved_idx = result.find("# Resolved Critiques")
    crit001_idx = result.find("CRIT-001")
    assert crit001_idx > resolved_idx

    # Unresolved count should drop
    count = count_unresolved_critiques(result)
    assert count == 1  # CRIT-002 still unresolved


def test_resolve_critique_already_resolved():
    text = (FIXTURES / "critique_log.md").read_text()
    # CRIT-003 is already resolved — should return text unchanged
    result = resolve_critique(text, "CRIT-003", "Duplicate resolution.")
    assert result == text


def test_resolve_critique_not_found():
    text = (FIXTURES / "critique_log.md").read_text()
    result = resolve_critique(text, "CRIT-999", "Not found.")
    assert result == text


def test_resolve_critique_with_phase_subheadings():
    """Resolving a critique with ### Phase 1/Phase 2 removes the entire block from Active."""
    text = """\
---
total_critiques: 2
unresolved_high: 1
unresolved_medium: 1
---

# Active Critiques

## CRIT-010 [HIGH] [UNRESOLVED]
- **Target:** WH-002
- **Filed:** iteration 3

### Phase 1: Reproduce
1. Start with metric ansatz
2. Apply field equations

### Phase 2: Objection
- **What is wrong:** Missing factor of 2 in normalization
- **Why it matters:** Changes final temperature by factor 2

## CRIT-011 [MEDIUM] [UNRESOLVED]
- **Target:** ER-001
- **Filed:** iteration 4
- **Critique:** Check boundary conditions.

# Resolved Critiques
"""
    result = resolve_critique(text, "CRIT-010", "Factor of 2 corrected.")

    # CRIT-010 should be in Resolved section
    resolved_idx = result.find("# Resolved Critiques")
    crit010_idx = result.find("CRIT-010")
    assert crit010_idx > resolved_idx

    # Phase 1/Phase 2 content must NOT remain in Active section
    active_idx = result.find("# Active Critiques")
    active_section = result[active_idx:resolved_idx]
    assert "Phase 1" not in active_section, "Phase 1 body orphaned in Active"
    assert "Phase 2" not in active_section, "Phase 2 body orphaned in Active"
    assert "metric ansatz" not in active_section
    assert "Missing factor of 2" not in active_section

    # Phase content SHOULD be in Resolved section
    resolved_section = result[resolved_idx:]
    assert "Phase 1" in resolved_section
    assert "Phase 2" in resolved_section
    assert "Factor of 2 corrected." in resolved_section

    # CRIT-011 should still be in Active
    assert "CRIT-011" in active_section
    count = count_unresolved_critiques(result)
    assert count == 1  # only CRIT-011 (MEDIUM) still unresolved


# --- Tests for filter_self_retracted_critiques ---


def test_filter_mixed_response():
    """One genuine MEDIUM + one self-retracted LOW → only MEDIUM survives."""
    text = """\
## CRIT-010 [MEDIUM] [UNRESOLVED]
- **Target:** WH-002
- **Filed:** iteration 3

### Phase 1: Reproduce
1. Start with metric ansatz
2. Apply field equations

### Phase 2: Objection
- **What is wrong:** Missing factor of 2 in normalization
- **Why it matters:** Changes final temperature by factor 2

## CRIT-011 [LOW] [UNRESOLVED]
- **Target:** WH-003
- **Filed:** iteration 3

### Phase 1: Reproduce
1. Start with Hawking formula
2. Derive entropy expression
3. Matches claimed result

### Phase 2: Objection
- Reproduction succeeded, no issues found. Documenting what was checked.
- **Suggested verification:** None needed.
"""
    filtered, retracted = filter_self_retracted_critiques(text)
    assert "CRIT-010" in filtered
    assert "CRIT-011" in filtered
    assert "[WITHDRAWN]" in filtered
    assert len(retracted) == 1
    assert "CRIT-011" in retracted[0]
    assert "WH-003" in retracted[0]


def test_filter_genuine_low_kept():
    """LOW with real objection (notation issue) → not filtered."""
    text = """\
## CRIT-012 [LOW] [UNRESOLVED]
- **Target:** WH-001
- **Filed:** iteration 2

### Phase 1: Reproduce
1. Follow the derivation step by step.

### Phase 2: Objection
- **What is wrong:** Inconsistent notation: kappa used for both surface gravity and a coupling constant.
- **Why it matters:** Could confuse later derivations.
- **Suggested verification:** Notation audit.
"""
    filtered, retracted = filter_self_retracted_critiques(text)
    assert "CRIT-012" in filtered
    assert retracted == []


def test_filter_all_retracted():
    """Two self-retracted LOWs → empty filtered text, two retracted summaries."""
    text = """\
## CRIT-013 [LOW] [UNRESOLVED]
- **Target:** WH-001
- **Filed:** iteration 4

### Phase 1: Reproduce
1. Verified all steps.

### Phase 2: Objection
- Reproduction succeeded, no issues found.
- **Suggested verification:** None needed.

## CRIT-014 [LOW] [UNRESOLVED]
- **Target:** ER-001
- **Filed:** iteration 4

### Phase 1: Reproduce
1. Checked the algebra.

### Phase 2: Objection
- Reproduction succeeded, no flaws detected. Documenting successful verification.
"""
    filtered, retracted = filter_self_retracted_critiques(text)
    assert "CRIT-013" in filtered
    assert "CRIT-014" in filtered
    assert "[WITHDRAWN]" in filtered
    assert "[UNRESOLVED]" not in filtered
    assert len(retracted) == 2


def test_filter_no_critiques_filed_marker():
    """NO_CRITIQUES_FILED marker → empty text, empty retracted list."""
    text = "NO_CRITIQUES_FILED: Reviewed 5 claims, no issues found."
    filtered, retracted = filter_self_retracted_critiques(text)
    assert filtered == ""
    assert retracted == []


def test_filter_high_never_filtered():
    """HIGH with retraction-like language → never suppressed."""
    text = """\
## CRIT-015 [HIGH] [UNRESOLVED]
- **Target:** WH-004
- **Filed:** iteration 5

### Phase 1: Reproduce
1. Attempted to reproduce the entropy bound derivation.

### Phase 2: Objection
- Reproduction succeeded, no issues found in algebra, BUT the boundary
  condition is applied at the wrong surface.
- **Why it matters:** Invalidates the area law derivation.
"""
    filtered, retracted = filter_self_retracted_critiques(text)
    assert "CRIT-015" in filtered
    assert retracted == []


def test_filter_critique_nnn_drift_tolerance():
    """CRITIQUE-NNN format (LLM drift) is also handled."""
    text = """\
## CRITIQUE-015 [LOW] [UNRESOLVED]
- **Target:** WH-005
- **Filed:** iteration 6

### Phase 1: Reproduce
1. Steps match claimed result.

### Phase 2: Objection
- Reproduction succeeded, no errors. Documenting successful verification.
- **Suggested verification:** None needed.
"""
    filtered, retracted = filter_self_retracted_critiques(text)
    assert "CRITIQUE-015" in filtered
    assert "[WITHDRAWN]" in filtered
    assert len(retracted) == 1
    assert "CRITIQUE-015" in retracted[0]


# --- Tests for recount_critique_metadata ---


def test_recount_critique_metadata():
    text = (FIXTURES / "critique_log.md").read_text()
    meta = recount_critique_metadata(text)
    assert meta["unresolved_critiques"] == 2  # 1 HIGH + 1 MEDIUM
    assert meta["total_critiques"] >= 3  # CRIT-001, CRIT-002, CRIT-003
    assert meta["withdrawn_critiques"] >= 0


def test_recount_critique_metadata_empty():
    meta = recount_critique_metadata("No critiques.")
    assert meta == {
        "unresolved_critiques": 0,
        "total_critiques": 0,
        "withdrawn_critiques": 0,
    }


# --- Tests for computation log parsing and stall detection ---

SAMPLE_COMP_LOG = """\
---
total_computations: 5
---

# Computations

## COMP-001: Check WH-001 energy
- **CLAIM**: Verify WH-001 ground state energy E_0 = hbar*omega/2
- **VERDICT**: VERIFIED
- **RESULT**:
  All checks passed. E_0 = 0.5 * hbar * omega.

## COMP-002: Check WH-002 partition function
- **CLAIM**: Verify WH-002 partition function Z = 1/(1 - exp(-beta*hbar*omega))
- **VERDICT**: INCONCLUSIVE
- **RESULT**:
  Symbolic simplification failed. Numerical checks inconclusive at high T.

## COMP-003: Retry WH-002 partition function
- **CLAIM**: Verify WH-002 partition function Z = 1/(1 - exp(-beta*hbar*omega))
- **VERDICT**: INCONCLUSIVE
- **RESULT**:
  Series expansion did not converge within 100 terms at beta=0.01.

## COMP-004: Retry WH-002 again
- **CLAIM**: Verify WH-002 partition function Z = 1/(1 - exp(-beta*hbar*omega))
- **VERDICT**: INCONCLUSIVE
- **RESULT**:
  Used 10000 terms but still 8% discrepancy at beta=0.001.

## COMP-005: Check ER-001 temperature
- **CLAIM**: Verify ER-001 Hawking temperature T_H = hbar*kappa/(2*pi)
- **VERDICT**: VERIFIED
- **RESULT**:
  All numerical checks passed.
"""


class TestParseCompEntries:
    def test_parse_comp_entries(self):
        entries = _parse_comp_entries(SAMPLE_COMP_LOG)
        assert len(entries) == 5
        assert entries[0]["id"] == "COMP-001"
        assert "WH-001" in entries[0]["claim"]
        assert entries[0]["verdict"] == "VERIFIED"
        assert "E_0 = 0.5" in entries[0]["result"]

        assert entries[1]["id"] == "COMP-002"
        assert entries[1]["verdict"] == "INCONCLUSIVE"

    def test_parse_comp_entries_empty(self):
        entries = _parse_comp_entries("No computations yet.")
        assert entries == []


class TestDetectStalls:
    def test_detect_stalls_triggered(self):
        """3 INCONCLUSIVE on same claim (WH-002) -> returns stall."""
        stalls = detect_computation_stalls(SAMPLE_COMP_LOG, threshold=3)
        assert len(stalls) == 1
        assert stalls[0]["count"] == 3
        assert "WH-002" in stalls[0]["claim"]
        assert all(v == "INCONCLUSIVE" for v in stalls[0]["verdicts"])

    def test_detect_stalls_verified_resets(self):
        """VERIFIED breaks the streak — WH-001 has VERIFIED so no stall."""
        stalls = detect_computation_stalls(SAMPLE_COMP_LOG, threshold=1)
        # WH-001 was VERIFIED, so no stall for it
        claim_keys = [s["claim"] for s in stalls]
        assert not any("WH-001" in k for k in claim_keys)

    def test_detect_stalls_below_threshold(self):
        """2 failures -> no stall at threshold=3."""
        log = """\
## COMP-001: Check claim
- **CLAIM**: Verify WH-003 something
- **VERDICT**: INCONCLUSIVE
- **RESULT**:
  Failed.

## COMP-002: Retry claim
- **CLAIM**: Verify WH-003 something else
- **VERDICT**: INCONCLUSIVE
- **RESULT**:
  Failed again.
"""
        stalls = detect_computation_stalls(log, threshold=3)
        assert stalls == []


class TestFindPriorFailures:
    def test_find_prior_failures_by_er_id(self):
        """Match via WH-002 in task body."""
        results = find_prior_failures_for_claim(
            SAMPLE_COMP_LOG, "Verify WH-002 partition function using numerical methods"
        )
        assert len(results) == 3  # COMP-002, COMP-003, COMP-004
        # Most recent first
        assert "8% discrepancy" in results[0]
        assert "100 terms" in results[1]

    def test_find_prior_failures_none(self):
        """No matching failures -> empty list."""
        results = find_prior_failures_for_claim(
            SAMPLE_COMP_LOG, "Verify WH-099 something totally unrelated"
        )
        assert results == []

    def test_find_prior_failures_skips_verified(self):
        """VERIFIED entries for the same claim are not returned."""
        results = find_prior_failures_for_claim(
            SAMPLE_COMP_LOG, "Check WH-001 ground state energy"
        )
        # COMP-001 is VERIFIED, should not appear
        assert results == []


class TestErWhSectionCounting:
    """Tests for robust ER/WH section detection across format variants."""

    def test_count_er_h2_headers(self):
        text = "## ER-001 Title\nBody\n\n## ER-002 Title\nBody\n"
        assert count_er_sections(text) == 2

    def test_count_er_bold_format(self):
        text = "**ER-001 — Partition Function Z**\nBody\n\n**ER-002 — Mean Energy**\nBody\n"
        assert count_er_sections(text) == 2

    def test_count_er_mixed_formats(self):
        text = "## ER-001 Title\nBody\n\n**ER-002 — Bold Title**\nBody\n"
        assert count_er_sections(text) == 2

    def test_count_wh_h2_headers(self):
        text = "## WH-001 Hypothesis\nBody\n\n## WH-002 Another\nBody\n"
        assert count_wh_sections(text) == 2

    def test_count_wh_bold_format(self):
        text = "**WH-001 — Some hypothesis**\nBody\n"
        assert count_wh_sections(text) == 1

    def test_count_ignores_inline_references(self):
        """ER-NNN in body text (not at line start as header/bold) should not be counted."""
        text = "## ER-001 Title\nThis depends on ER-002.\n"
        assert count_er_sections(text) == 1

    def test_find_er_ids_mixed(self):
        text = "## ER-001 Title\nBody\n\n**ER-003 — Bold**\nBody\n"
        ids = find_er_section_ids(text)
        assert ids == ["ER-001", "ER-003"]

    def test_count_empty_text(self):
        assert count_er_sections("") == 0
        assert count_wh_sections("") == 0

    def test_h3_headers_also_counted(self):
        """H3 headers (### ER-NNN) should also be detected."""
        text = "### ER-001 Schwarzschild metric\nContent\n"
        assert count_er_sections(text) == 1


class TestNormalizeErWhHeaders:
    """Tests for the bold-to-H2 normalizer."""

    def test_converts_bold_er_to_h2(self):
        text = "**ER-001 — Partition Function Z**\nBody text.\n"
        result = normalize_er_wh_headers(text)
        assert result == "## ER-001 — Partition Function Z\nBody text.\n"

    def test_converts_bold_wh_to_h2(self):
        text = "**WH-003 — Some hypothesis**\nBody.\n"
        result = normalize_er_wh_headers(text)
        assert result == "## WH-003 — Some hypothesis\nBody.\n"

    def test_leaves_h2_headers_unchanged(self):
        text = "## ER-001 Title\nBody.\n"
        assert normalize_er_wh_headers(text) == text

    def test_converts_multiple_entries(self):
        text = (
            "**ER-001 — First**\nBody 1.\n\n"
            "**ER-002 — Second**\nBody 2.\n\n"
            "**WH-001 — Hypothesis**\nBody 3.\n"
        )
        result = normalize_er_wh_headers(text)
        assert "## ER-001 — First" in result
        assert "## ER-002 — Second" in result
        assert "## WH-001 — Hypothesis" in result
        assert "**ER-" not in result
        assert "**WH-" not in result

    def test_preserves_non_er_wh_bold(self):
        """Bold text that isn't ER/WH should be untouched."""
        text = "**Important note**\nSome text.\n\n**ER-001 — Title**\nBody.\n"
        result = normalize_er_wh_headers(text)
        assert "**Important note**" in result
        assert "## ER-001 — Title" in result


# ---------------------------------------------------------------------------
# Regression tests for code-fenced frontmatter and colon-inside-bold format
# ---------------------------------------------------------------------------


class TestFrontmatterCodeFenced:
    """parse_frontmatter must handle LLM-emitted code fences around YAML."""

    def test_code_fenced_yaml(self):
        text = "```yaml\n---\ntask_type: terminate\nassigned_to: orchestrator\n---\n```\n\nBody."
        meta, body = parse_frontmatter(text)
        assert meta["task_type"] == "terminate"
        assert meta["assigned_to"] == "orchestrator"
        assert "Body." in body

    def test_code_fenced_no_lang(self):
        text = "```\n---\nstatus: completed\n---\n```\n\nDone."
        meta, body = parse_frontmatter(text)
        assert meta["status"] == "completed"

    def test_bare_frontmatter_still_works(self):
        text = "---\ntitle: hello\n---\n\nBody."
        meta, body = parse_frontmatter(text)
        assert meta["title"] == "hello"


class TestVerdictColonInsideBold:
    """_parse_comp_entries must parse **VERDICT:** (colon inside bold markers)."""

    def test_verdict_colon_inside_bold(self):
        log = """\
## COMP-001: Verify QHO

**CLAIM:** Partition function matches.
- WH-001: Z = exp(-x/2)/(1-exp(-x))

**RESULT:**
All checks pass.

**VERDICT:** VERIFIED for WH-001.
"""
        entries = _parse_comp_entries(log)
        assert len(entries) == 1
        assert entries[0]["verdict"] == "VERIFIED"

    def test_verdict_colon_outside_bold(self):
        """Original format still works."""
        log = """\
## COMP-001: Check
- **CLAIM**: Verify WH-001
- **VERDICT**: VERIFIED
- **RESULT**:
  Done.
"""
        entries = _parse_comp_entries(log)
        assert entries[0]["verdict"] == "VERIFIED"

    def test_claim_colon_inside_bold(self):
        log = """\
## COMP-001: Check

**CLAIM:** Four working hypotheses for QHO thermodynamics:
- WH-001: Z = exp(-x/2)/(1-exp(-x))

**VERDICT:** VERIFIED
"""
        entries = _parse_comp_entries(log)
        assert "Four working hypotheses" in entries[0]["claim"]
        # claim should not start with residual **
        assert not entries[0]["claim"].startswith("**")

    def test_body_field_contains_full_text(self):
        """Entries include a body field for searching WH/ER IDs beyond the claim line."""
        log = """\
## COMP-001: Verification

**CLAIM:** Four working hypotheses for QHO:
- WH-001: partition function
- WH-002: mean energy

**VERDICT:** VERIFIED
"""
        entries = _parse_comp_entries(log)
        assert "WH-001" in entries[0]["body"]
        assert "WH-002" in entries[0]["body"]


# ---------------------------------------------------------------------------
# Improvement 5: WITHDRAWN critique status
# ---------------------------------------------------------------------------


class TestWithdrawnStatus:
    """Tests for WITHDRAWN critique status (Improvement 5)."""

    def test_filter_medium_self_retracted(self):
        """MEDIUM with retraction signals -> WITHDRAWN."""
        text = """\
## CRIT-020 [MEDIUM] [UNRESOLVED]
- **Target:** WH-005
- **Filed:** iteration 6

### Phase 1: Reproduce
1. Steps all check out.

### Phase 2: Objection
- No flaws found in the derivation. The algebra is correct.
- **Suggested verification:** None needed.
"""
        filtered, retracted = filter_self_retracted_critiques(text)
        assert "CRIT-020" in filtered
        assert "[WITHDRAWN]" in filtered
        assert "[UNRESOLVED]" not in filtered
        assert len(retracted) == 1
        assert "CRIT-020" in retracted[0]

    def test_withdrawn_not_counted_unresolved(self):
        """WITHDRAWN critiques should not count as unresolved."""
        text = "## CRIT-020 [MEDIUM] [WITHDRAWN]\n- **Target:** WH-005\n"
        count = count_unresolved_critiques(text)
        assert count == 0

    def test_withdrawn_counted_in_total(self):
        """WITHDRAWN critiques should count in total_critiques."""
        text = "## CRIT-020 [MEDIUM] [WITHDRAWN]\n- **Target:** WH-005\n"
        from open_dirac.utils.markdown import recount_critique_metadata

        meta = recount_critique_metadata(text)
        assert meta["total_critiques"] == 1
        assert meta["withdrawn_critiques"] == 1

    def test_filter_medium_genuine_kept(self):
        """MEDIUM with real objection -> NOT withdrawn."""
        text = """\
## CRIT-021 [MEDIUM] [UNRESOLVED]
- **Target:** WH-006
- **Filed:** iteration 7

### Phase 1: Reproduce
1. Follow the derivation.

### Phase 2: Objection
- **What is wrong:** Sign error in step 3 of the entropy calculation.
- **Why it matters:** Could change the final result by a factor of -1.
"""
        filtered, retracted = filter_self_retracted_critiques(text)
        assert "CRIT-021" in filtered
        assert "[UNRESOLVED]" in filtered
        assert "[WITHDRAWN]" not in filtered
        assert retracted == []

    def test_high_never_withdrawn(self):
        """HIGH with retraction language -> never auto-retracted."""
        text = """\
## CRIT-022 [HIGH] [UNRESOLVED]
- **Target:** WH-007
- **Filed:** iteration 8

### Phase 1: Reproduce
1. Checked every step.

### Phase 2: Objection
- No flaws found in the algebra, BUT boundary condition wrong.
- **Why it matters:** Invalidates the result.
"""
        filtered, retracted = filter_self_retracted_critiques(text)
        assert "CRIT-022" in filtered
        assert "[UNRESOLVED]" in filtered
        assert retracted == []

    def test_new_retraction_patterns(self):
        """New retraction patterns are detected."""
        cases = [
            "No flaw found in the derivation.",
            "No flaws identified during review.",
            "This does not warrant filing a critique.",
            "Not filing any critique.",
            "No genuine objections to file.",
            "No issues to raise against this claim.",
        ]
        for case in cases:
            text = f"""\
## CRIT-099 [LOW] [UNRESOLVED]
- **Target:** WH-099
- **Filed:** iteration 1

### Phase 2: Objection
- {case}
"""
            filtered, retracted = filter_self_retracted_critiques(text)
            assert len(retracted) == 1, f"Pattern not detected: {case}"

    def test_withdrawn_metadata_counting(self):
        """count_withdrawn_critiques returns correct count."""
        text = """\
## CRIT-001 [HIGH] [UNRESOLVED]
- Active critique.

## CRIT-002 [MEDIUM] [WITHDRAWN]
- Withdrawn critique.

## CRIT-003 [LOW] [WITHDRAWN]
- Another withdrawn critique.
"""
        assert count_withdrawn_critiques(text) == 2
        count = count_unresolved_critiques(text)
        assert count == 1  # only CRIT-001 (HIGH) is unresolved


# ---------------------------------------------------------------------------
# Improvement 2A: Flatten nested unverified brackets
# ---------------------------------------------------------------------------


class TestFlattenUnverifiedBrackets:
    """Tests for nested bracket flattening (Improvement 2A)."""

    def test_flatten_single(self):
        text = "[COMP-001:unverified]"
        assert flatten_unverified_brackets(text) == "[COMP-001:unverified]"

    def test_flatten_double_nested(self):
        text = "[[COMP-001:unverified]:unverified]"
        assert flatten_unverified_brackets(text) == "[COMP-001:unverified]"

    def test_flatten_triple(self):
        text = "[[[COMP-001:unverified]:unverified]:unverified]"
        result = flatten_unverified_brackets(text)
        assert result == "[COMP-001:unverified]"

    def test_flatten_mixed(self):
        text = "See [[COMP-001:unverified]:unverified] and [TASK-002:unverified] for details."
        result = flatten_unverified_brackets(text)
        assert "[COMP-001:unverified]" in result
        assert "[TASK-002:unverified]" in result
        assert "[[" not in result

    def test_flatten_idempotent(self):
        text = "[COMP-001:unverified]"
        assert flatten_unverified_brackets(flatten_unverified_brackets(text)) == text

    def test_flatten_no_markers(self):
        text = "Regular text with no markers."
        assert flatten_unverified_brackets(text) == text


# ---------------------------------------------------------------------------
# Improvement 1D: Zero-output stall detection
# ---------------------------------------------------------------------------


class TestDetectZeroOutputStalls:
    """Tests for zero-output stall detection (Improvement 1D)."""

    def test_detect_zero_output_stalls(self):
        """Entries with 'Agent produced no text output' are detected."""
        log = """\
## COMP-001: Check WH-001
- **CLAIM**: Verify WH-001 formula
- **VERDICT**: INCONCLUSIVE

Agent produced no text output. Writing INCONCLUSIVE stub.
"""
        stalls = detect_zero_output_stalls(log)
        assert len(stalls) == 1
        assert "WH-001" in stalls[0]["claim"]
        assert stalls[0]["entry_id"] == "COMP-001"

    def test_detect_zero_output_stalls_normal(self):
        """Normal entries are not flagged."""
        log = """\
## COMP-001: Check WH-001
- **CLAIM**: Verify WH-001 formula
- **VERDICT**: VERIFIED
- **RESULT**:
  All checks pass.
"""
        stalls = detect_zero_output_stalls(log)
        assert stalls == []


class TestMethodNotesExtraction:
    """Tests for METHOD/NOTES extraction in _parse_comp_entries and failure excerpts."""

    COMP_LOG_WITH_METHOD_NOTES = """\
## COMP-001: Check WH-001
**CLAIM**: Verify WH-001 ground state energy
**METHOD**:
Used numerical integration with x0=1e-6 grid spacing.
Scipy quad integration over [0, 10].
**VERDICT**: REFUTED
**RESULT**:
Got E_0 = 0.48, expected 0.50. 4% discrepancy.
**NOTES**:
The x0=1e-6 starting point may introduce boundary effects.
Consider larger integration domain.
"""

    def test_parse_comp_entries_extracts_method_and_notes(self):
        """_parse_comp_entries populates method and notes fields."""
        entries = _parse_comp_entries(self.COMP_LOG_WITH_METHOD_NOTES)
        assert len(entries) == 1
        entry = entries[0]
        assert "numerical integration" in entry["method"]
        assert "x0=1e-6" in entry["method"]
        assert "boundary effects" in entry["notes"]
        assert entry["verdict"] == "REFUTED"
        assert "E_0 = 0.48" in entry["result"]

    def test_find_prior_failures_includes_method(self):
        """find_prior_failures_for_claim returns excerpts containing METHOD text."""
        results = find_prior_failures_for_claim(
            self.COMP_LOG_WITH_METHOD_NOTES, "Verify WH-001 ground state energy"
        )
        assert len(results) == 1
        assert "numerical integration" in results[0]
        assert "Method" in results[0] or "method" in results[0].lower()

    def test_find_prior_failures_includes_notes(self):
        """find_prior_failures_for_claim returns excerpts containing NOTES text."""
        results = find_prior_failures_for_claim(
            self.COMP_LOG_WITH_METHOD_NOTES, "Verify WH-001 ground state energy"
        )
        assert len(results) == 1
        assert "boundary effects" in results[0]

    def test_find_prior_failures_no_result_block(self):
        """Entries with METHOD but empty RESULT still produce excerpts."""
        log = """\
## COMP-001: Check WH-002
**CLAIM**: Verify WH-002 partition function
**METHOD**:
Taylor expansion to 50 terms at beta=0.1.
**VERDICT**: INCONCLUSIVE
"""
        results = find_prior_failures_for_claim(log, "Verify WH-002 partition function")
        assert len(results) == 1
        assert "Taylor expansion" in results[0]
        assert "INCONCLUSIVE" in results[0]

    def test_format_failure_excerpt_all_fields(self):
        """_format_failure_excerpt combines all available fields."""
        entry = {
            "verdict": "REFUTED",
            "method": "Direct numerical integration",
            "result": "Got 0.48 instead of 0.50",
            "notes": "Grid too coarse",
        }
        excerpt = _format_failure_excerpt(entry)
        assert "REFUTED" in excerpt
        assert "Direct numerical integration" in excerpt
        assert "Got 0.48" in excerpt
        assert "Grid too coarse" in excerpt

    def test_format_failure_excerpt_partial_fields(self):
        """_format_failure_excerpt works with only some fields populated."""
        entry = {
            "verdict": "INCONCLUSIVE",
            "method": "",
            "result": "Failed",
            "notes": "",
        }
        excerpt = _format_failure_excerpt(entry)
        assert "INCONCLUSIVE" in excerpt
        assert "Failed" in excerpt
        assert "Method" not in excerpt  # empty method not included
