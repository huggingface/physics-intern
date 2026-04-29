"""Tests for the formal ResearchState module."""

import json

from open_dirac.research_state import (
    ResearchState,
    Hypothesis,
    HypothesisStatus,
    Evidence,
    ReviewResult,
    Verdict,
    Critique,
    Severity,
    CritiqueStatus,
    FailedApproach,
    _extract_hypothesis_sections,
    _extract_h1_section,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESEARCH_STATE = """\
---
problem_id: research-session
title: Test problem
status: not_started
iteration: 5
---

# Problem Statement

Some problem.

# Working Hypotheses (WH) and Established Results (ER)

## WH-001 — First hypothesis

The photon has spin-1.

Some derivation details here.

## ER-002 — Verified result

Energy is conserved.

This has been verified.

## WH-003 — Third hypothesis

Entropy increases.

# Dead Ends

Nothing yet.
"""


# ---------------------------------------------------------------------------
# Hypothesis section extraction
# ---------------------------------------------------------------------------


class TestExtractHypothesisSections:
    def test_extracts_wh_and_er(self):
        sections = _extract_hypothesis_sections(SAMPLE_RESEARCH_STATE)
        ids = [s[0] for s in sections]
        assert ids == ["WH-001", "ER-002", "WH-003"]

    def test_title_extracted(self):
        sections = _extract_hypothesis_sections(SAMPLE_RESEARCH_STATE)
        assert sections[0][1] == "First hypothesis"
        assert sections[1][1] == "Verified result"

    def test_body_contains_content(self):
        sections = _extract_hypothesis_sections(SAMPLE_RESEARCH_STATE)
        assert "photon has spin-1" in sections[0][2]

    def test_empty_body(self):
        sections = _extract_hypothesis_sections("No sections here.")
        assert sections == []

    def test_section_body_stops_at_h1(self):
        sections = _extract_hypothesis_sections(SAMPLE_RESEARCH_STATE)
        # WH-003's body should not contain "# Dead Ends"
        assert "Dead Ends" not in sections[2][2]


# ---------------------------------------------------------------------------
# Query methods
# ---------------------------------------------------------------------------


class TestQueryMethods:
    def _make_state(self) -> ResearchState:
        state = ResearchState(iteration=5)
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="Test",
            status=HypothesisStatus.WORKING,
            critiques=["CRIT-001"],
            evidence=[Evidence(type="compute", method="numerical", result="42")],
        )
        state.hypotheses["ER-002"] = Hypothesis(
            id="ER-002",
            statement="Verified",
            status=HypothesisStatus.ESTABLISHED,
            evidence=[Evidence(type="compute", method="symbolic", result="ok")],
            review=ReviewResult(verdict=Verdict.VERIFIED, summary="Confirmed."),
        )
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.HIGH,
            status=CritiqueStatus.ACTIVE,
        )
        return state

    def test_has_verified_evidence(self):
        state = self._make_state()
        assert state.has_verified_evidence("ER-002")
        assert not state.has_verified_evidence("WH-001")

    def test_hypotheses_with_evidence(self):
        state = self._make_state()
        with_ev = state.hypotheses_with_evidence()
        assert len(with_ev) == 2
        ids = {h.id for h in with_ev}
        assert ids == {"WH-001", "ER-002"}

    def test_active_critiques_for(self):
        state = self._make_state()
        assert len(state.active_critiques_for("WH-001")) == 1
        assert len(state.active_critiques_for("ER-002")) == 0

    def test_established_hypotheses(self):
        state = self._make_state()
        assert len(state.established_hypotheses()) == 1
        assert state.established_hypotheses()[0].id == "ER-002"

    def test_working_hypotheses(self):
        state = self._make_state()
        assert len(state.working_hypotheses()) == 1
        assert state.working_hypotheses()[0].id == "WH-001"


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_json_round_trip(self):
        state = ResearchState(iteration=3)
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001", statement="Test", status=HypothesisStatus.WORKING
        )
        state.hypotheses["WH-001"].evidence = [
            Evidence(
                type="compute",
                method="numerical",
                result="42",
                confidence="exact",
                iteration=2,
            )
        ]
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001", targets=["WH-001"], severity=Severity.HIGH
        )
        state.failed_approaches.append(
            FailedApproach(
                description="Tried perturbation theory",
                reason="Divergent series",
                related_entities=["WH-001"],
                iteration=2,
            )
        )

        json_str = state.to_json()
        restored = ResearchState.from_json(json_str)

        assert restored.iteration == 3
        assert "WH-001" in restored.hypotheses
        assert restored.hypotheses["WH-001"].statement == "Test"
        assert len(restored.hypotheses["WH-001"].evidence) > 0
        assert restored.hypotheses["WH-001"].evidence[0].result == "42"
        assert restored.critiques["CRIT-001"].severity == Severity.HIGH
        assert len(restored.failed_approaches) == 1
        assert restored.failed_approaches[0].description == "Tried perturbation theory"

    def test_save_and_load(self, tmp_path):
        state = ResearchState(iteration=7)
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001", statement="E=mc2", status=HypothesisStatus.ESTABLISHED
        )
        state.save(tmp_path)

        loaded = ResearchState.load(tmp_path)
        assert loaded.iteration == 7
        assert "ER-001" in loaded.hypotheses
        assert loaded.hypotheses["ER-001"].status == HypothesisStatus.ESTABLISHED

    def test_load_missing_file(self, tmp_path):
        loaded = ResearchState.load(tmp_path)
        assert loaded.iteration == 0
        assert loaded.hypotheses == {}

    def test_json_is_valid(self):
        state = ResearchState(iteration=1)
        state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        parsed = json.loads(state.to_json())
        assert parsed["iteration"] == 1
        assert "WH-001" in parsed["hypotheses"]


# ---------------------------------------------------------------------------
# Phase 4: Failure tracking
# ---------------------------------------------------------------------------


class TestFailureTracking:
    def test_failures_for_hypothesis(self):
        state = ResearchState()
        state.failed_approaches.append(
            FailedApproach(
                description="REFUTED on: WH-001 spin prediction",
                reason="Got spin-0 instead of spin-1",
                related_entities=["WH-001"],
                iteration=3,
            )
        )
        state.failed_approaches.append(
            FailedApproach(
                description="INCONCLUSIVE on: WH-002 entropy",
                reason="Timeout",
                related_entities=["WH-002"],
                iteration=4,
            )
        )
        wh1_failures = state.failures_for_hypothesis("WH-001")
        assert len(wh1_failures) == 1
        assert "spin prediction" in wh1_failures[0].description

    def test_failures_round_trip(self):
        state = ResearchState(iteration=5)
        state.failed_approaches.append(
            FailedApproach(
                description="REFUTED on: WH-001",
                reason="Wrong coefficient",
                related_entities=["WH-001"],
                iteration=4,
            )
        )
        restored = ResearchState.from_json(state.to_json())
        assert len(restored.failed_approaches) == 1
        assert restored.failed_approaches[0].reason == "Wrong coefficient"
        assert restored.failed_approaches[0].related_entities == ["WH-001"]

    def test_no_failures_returns_empty(self):
        state = ResearchState()
        assert state.failures_for_hypothesis("WH-001") == []

    def test_abandoned_hypotheses_query(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.ABANDONED,
        )
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            status=HypothesisStatus.WORKING,
        )
        state.hypotheses["ER-003"] = Hypothesis(
            id="ER-003",
            status=HypothesisStatus.ESTABLISHED,
        )
        assert len(state.abandoned_hypotheses()) == 1
        assert state.abandoned_hypotheses()[0].id == "WH-001"

    def test_abandoned_status_round_trip(self):
        state = ResearchState(iteration=5)
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="Bad idea",
            status=HypothesisStatus.ABANDONED,
            iteration_modified=3,
        )
        restored = ResearchState.from_json(state.to_json())
        assert restored.hypotheses["WH-001"].status == HypothesisStatus.ABANDONED
        assert restored.hypotheses["WH-001"].iteration_modified == 3


# ---------------------------------------------------------------------------
# New ResearchState fields
# ---------------------------------------------------------------------------


class TestNewResearchStateFields:
    def test_defaults(self):
        state = ResearchState()
        assert state.problem_statement == ""
        assert state.conventions == ""
        assert state.strategy == ""
        assert state.research_notes == []
        assert state.status == "in_progress"
        assert state.title == ""
        assert state.survey_background == ""
        assert state.survey_methods == ""
        assert state.known_pitfalls == ""

    def test_json_round_trip(self):
        state = ResearchState(
            iteration=3,
            problem_statement="Derive Hawking temperature.",
            conventions="Natural units: hbar = c = k_B = 1.",
            strategy="Focus on surface gravity approach.",
            research_notes=[{"text": "Surface gravity confirmed.", "iteration": 1}],
            status="complete",
            title="Hawking Temperature",
        )
        restored = ResearchState.from_json(state.to_json())
        assert restored.problem_statement == "Derive Hawking temperature."
        assert restored.conventions == "Natural units: hbar = c = k_B = 1."
        assert restored.strategy == "Focus on surface gravity approach."
        assert restored.research_notes == [
            {"text": "Surface gravity confirmed.", "iteration": 1}
        ]
        assert restored.status == "complete"
        assert restored.title == "Hawking Temperature"

    def test_backward_compat_missing_fields(self):
        """Loading old JSON without new fields should use defaults."""
        old_json = json.dumps(
            {
                "iteration": 5,
                "hypotheses": {},
                "critiques": {},
                "failed_approaches": [],
            }
        )
        state = ResearchState.from_json(old_json)
        assert state.iteration == 5
        assert state.problem_statement == ""
        assert state.conventions == ""
        assert state.status == "in_progress"
        assert state.title == ""
        assert state.survey_background == ""

    def test_backward_compat_open_questions_ignored(self):
        """Old JSON with open_questions field loads fine (silently ignored)."""
        old_json = json.dumps(
            {
                "iteration": 3,
                "open_questions": "What about grey-body factors?",
                "hypotheses": {},
                "critiques": {},
                "failed_approaches": [],
            }
        )
        state = ResearchState.from_json(old_json)
        assert state.iteration == 3


class TestSurveyFieldSerialization:
    def test_survey_fields_default_empty(self):
        state = ResearchState()
        assert state.survey_background == ""
        assert state.survey_methods == ""
        assert state.known_pitfalls == ""

    def test_survey_fields_json_round_trip(self):
        state = ResearchState(iteration=1)
        state.survey_background = "Hawking temperature via surface gravity."
        state.key_insights = "Use Killing vector method."
        state.survey_methods = "Euclidean path integral, Bogoliubov transformations"
        state.known_pitfalls = "Sign conventions for metric signature"
        json_str = state.to_json()
        restored = ResearchState.from_json(json_str)
        assert "Hawking temperature" in restored.survey_background
        assert "Killing vector" in restored.key_insights
        assert "Euclidean path integral" in restored.survey_methods
        assert "Sign conventions" in restored.known_pitfalls

    def test_missing_survey_fields_load_as_empty(self):
        """JSON without survey fields loads fine with empty defaults."""
        old_json = json.dumps(
            {
                "iteration": 5,
                "hypotheses": {},
                "critiques": {},
                "failed_approaches": [],
            }
        )
        state = ResearchState.from_json(old_json)
        assert state.survey_background == ""
        assert state.survey_methods == ""
        assert state.known_pitfalls == ""

    def test_problem_summary_round_trip(self):
        state = ResearchState(iteration=1)
        state.problem_summary = (
            "Derive the Hawking temperature for a Schwarzschild black hole."
        )
        json_str = state.to_json()
        restored = ResearchState.from_json(json_str)
        assert (
            restored.problem_summary
            == "Derive the Hawking temperature for a Schwarzschild black hole."
        )

    def test_missing_problem_summary_loads_as_empty(self):
        """JSON without problem_summary loads fine with empty default."""
        old_json = json.dumps(
            {"iteration": 5, "hypotheses": {}, "critiques": {}, "failed_approaches": []}
        )
        state = ResearchState.from_json(old_json)
        assert state.problem_summary == ""

    def test_survey_fields_serialize_as_strings(self):
        state = ResearchState()
        data = json.loads(state.to_json())
        assert data["survey_background"] == ""
        assert data["survey_methods"] == ""
        assert data["known_pitfalls"] == ""


# ---------------------------------------------------------------------------
# Evidence and ReviewResult on Hypothesis
# ---------------------------------------------------------------------------


class TestEvidenceOnHypothesis:
    def test_evidence_defaults(self):
        ev = Evidence()
        assert ev.type == ""
        assert ev.reasoning == ""
        assert ev.approach == ""
        assert ev.scripts == []
        assert ev.output == ""
        assert ev.method == ""
        assert ev.result == ""
        assert ev.confidence == ""
        assert ev.summary == ""
        assert ev.iteration is None

    def test_evidence_json_round_trip(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            evidence=[
                Evidence(
                    type="compute",
                    approach="Direct numerical computation",
                    scripts=["001_verify.py"],
                    output="Result: 0.785",
                    method="numerical integration",
                    result="pi/4 ~ 0.785",
                    confidence="approximate",
                    iteration=3,
                )
            ],
        )
        restored = ResearchState.from_json(state.to_json())
        ev_list = restored.hypotheses["WH-001"].evidence
        assert len(ev_list) > 0
        ev = ev_list[0]
        assert ev.type == "compute"
        assert ev.approach == "Direct numerical computation"
        assert ev.scripts == ["001_verify.py"]
        assert ev.method == "numerical integration"
        assert ev.result == "pi/4 ~ 0.785"
        assert ev.confidence == "approximate"
        assert ev.iteration == 3

    def test_evidence_summary_round_trip(self):
        """Evidence.summary survives JSON serialization."""
        state = ResearchState()
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            evidence=[
                Evidence(
                    type="research",
                    reasoning="Full derivation...",
                    method="contour integration",
                    result="I = 2*pi*i",
                    confidence="exact",
                    summary="Residue theorem gives I = 2*pi*i",
                    iteration=2,
                )
            ],
        )
        restored = ResearchState.from_json(state.to_json())
        ev_list = restored.hypotheses["WH-002"].evidence
        assert len(ev_list) > 0
        ev = ev_list[0]
        assert ev.summary == "Residue theorem gives I = 2*pi*i"
        assert ev.reasoning == "Full derivation..."

    def test_evidence_summary_on_rq_round_trip(self):
        """Evidence.summary on RQ survives JSON serialization."""
        from open_dirac.research_state import ResearchQuestion

        state = ResearchState()
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="What is X?",
            evidence=[
                Evidence(
                    type="research",
                    method="analysis",
                    result="X = 42",
                    confidence="exact",
                    summary="Found X by direct calculation",
                    iteration=1,
                )
            ],
        )
        restored = ResearchState.from_json(state.to_json())
        ev_list = restored.research_questions["RQ-001"].evidence
        assert len(ev_list) > 0
        ev = ev_list[0]
        assert ev.summary == "Found X by direct calculation"

    def test_review_result_defaults(self):
        vr = ReviewResult()
        assert vr.verdict == ""
        assert vr.summary == ""
        assert vr.iteration is None

    def test_review_result_json_round_trip(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            review=ReviewResult(
                verdict="VERIFIED",
                summary="All checks pass.",
                details="Minor notation issue noted but not blocking.",
                iteration=5,
            ),
        )
        restored = ResearchState.from_json(state.to_json())
        vr = restored.hypotheses["WH-001"].review
        assert vr is not None
        assert vr.verdict == "VERIFIED"
        assert vr.summary == "All checks pass."
        assert vr.details == "Minor notation issue noted but not blocking."
        assert vr.iteration == 5

    def test_refuted_count_json_round_trip(self):
        """refuted_count on Hypothesis survives serialization."""
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            refuted_count=2,
            review=ReviewResult(verdict="REFUTED", summary="Wrong", iteration=3),
        )
        restored = ResearchState.from_json(state.to_json())
        assert restored.hypotheses["WH-001"].refuted_count == 2

    def test_refuted_count_backward_compat(self):
        """Old JSON without refuted_count defaults to 0."""
        old_json = json.dumps(
            {
                "iteration": 1,
                "hypotheses": {
                    "WH-001": {"id": "WH-001", "status": "working"},
                },
            }
        )
        restored = ResearchState.from_json(old_json)
        assert restored.hypotheses["WH-001"].refuted_count == 0

    def test_hypothesis_without_evidence_or_review(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        restored = ResearchState.from_json(state.to_json())
        assert not restored.hypotheses["WH-001"].evidence
        assert restored.hypotheses["WH-001"].review is None

    def test_backward_compat_missing_evidence_fields(self):
        """Old JSON without evidence/review on hypotheses uses empty list / None."""
        old_json = json.dumps(
            {
                "iteration": 1,
                "hypotheses": {
                    "WH-001": {
                        "id": "WH-001",
                        "statement": "Test hypothesis",
                        "status": "working",
                    }
                },
                "critiques": {},
                "failed_approaches": [],
            }
        )
        state = ResearchState.from_json(old_json)
        h = state.hypotheses["WH-001"]
        assert not h.evidence
        assert h.review is None


# ---------------------------------------------------------------------------
# has_verified_evidence and hypotheses_with_evidence
# ---------------------------------------------------------------------------


class TestNewQueryMethods:
    def test_has_verified_evidence_true(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            review=ReviewResult(verdict=Verdict.VERIFIED),
        )
        assert state.has_verified_evidence("WH-001") is True

    def test_has_verified_evidence_false_no_verification(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        assert state.has_verified_evidence("WH-001") is False

    def test_has_verified_evidence_false_inconclusive(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            review=ReviewResult(verdict=Verdict.INCONCLUSIVE),
        )
        assert state.has_verified_evidence("WH-001") is False

    def test_has_verified_evidence_false_missing(self):
        state = ResearchState()
        assert state.has_verified_evidence("WH-999") is False

    def test_hypotheses_with_evidence_returns_correct(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            evidence=[Evidence(type="compute", result="42")],
        )
        state.hypotheses["WH-002"] = Hypothesis(id="WH-002")
        state.hypotheses["WH-003"] = Hypothesis(
            id="WH-003",
            evidence=[Evidence(type="research", result="derivation ok")],
        )
        result = state.hypotheses_with_evidence()
        assert len(result) == 2
        ids = {h.id for h in result}
        assert ids == {"WH-001", "WH-003"}

    def test_hypotheses_with_evidence_empty(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        assert state.hypotheses_with_evidence() == []


# ---------------------------------------------------------------------------
# next_*_num methods
# ---------------------------------------------------------------------------


class TestNextNumMethods:
    def test_next_hypothesis_num_basic(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        state.hypotheses["ER-003"] = Hypothesis(id="ER-003")
        assert state.next_hypothesis_num() == 4

    def test_next_hypothesis_num_empty(self):
        state = ResearchState()
        assert state.next_hypothesis_num() == 1

    def test_next_hypothesis_num_gap(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        state.hypotheses["WH-005"] = Hypothesis(id="WH-005")
        assert state.next_hypothesis_num() == 6

    def test_next_critique_num_basic(self):
        state = ResearchState()
        state.critiques["CRIT-001"] = Critique(id="CRIT-001")
        state.critiques["CRIT-002"] = Critique(id="CRIT-002")
        assert state.next_critique_num() == 3

    def test_next_critique_num_empty(self):
        state = ResearchState()
        assert state.next_critique_num() == 1

    def test_next_critique_num_non_sequential(self):
        state = ResearchState()
        state.critiques["CRIT-003"] = Critique(id="CRIT-003")
        state.critiques["CRIT-007"] = Critique(id="CRIT-007")
        assert state.next_critique_num() == 8


# ---------------------------------------------------------------------------
# _extract_h1_section
# ---------------------------------------------------------------------------


class TestExtractH1Section:
    MULTI_SECTION_DOC = """\
# Problem Statement

Derive the Hawking temperature from first principles.

# Conventions

Natural units: hbar = c = k_B = 1.

# Working Hypotheses (WH) and Established Results (ER)

## WH-001 — Something

Details here.

# Open Questions

What about grey-body factors?

# Dead Ends

Nothing yet.
"""

    def test_extracts_problem_statement(self):
        result = _extract_h1_section(self.MULTI_SECTION_DOC, "Problem Statement")
        assert "Hawking temperature" in result
        assert "Conventions" not in result

    def test_extracts_conventions(self):
        result = _extract_h1_section(self.MULTI_SECTION_DOC, "Conventions")
        assert "Natural units" in result
        assert "Working Hypotheses" not in result

    def test_extracts_open_questions(self):
        result = _extract_h1_section(self.MULTI_SECTION_DOC, "Open Questions")
        assert "grey-body factors" in result
        assert "Dead Ends" not in result

    def test_extracts_last_section(self):
        result = _extract_h1_section(self.MULTI_SECTION_DOC, "Dead Ends")
        assert "Nothing yet." in result

    def test_missing_section_returns_empty(self):
        result = _extract_h1_section(self.MULTI_SECTION_DOC, "Nonexistent")
        assert result == ""

    def test_empty_body_returns_empty(self):
        result = _extract_h1_section("", "Problem Statement")
        assert result == ""

    def test_single_section(self):
        doc = "# Problem Statement\n\nJust one section."
        result = _extract_h1_section(doc, "Problem Statement")
        assert result == "Just one section."


# ---------------------------------------------------------------------------
# Phase 3: Justification graph (B4) tests
# ---------------------------------------------------------------------------


class TestHypothesisDependsOn:
    """Tests for depends_on field."""

    def test_json_round_trip_new_fields(self):
        """depends_on survives JSON serialization round-trip."""
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="A depends on B",
            depends_on=["ER-001", "WH-002"],
        )
        json_str = state.to_json()
        restored = ResearchState.from_json(json_str)

        assert restored.hypotheses["WH-001"].depends_on == ["ER-001", "WH-002"]

    def test_json_backward_compat_missing_fields(self):
        """Old JSON without depends_on loads with defaults."""
        data = {
            "hypotheses": {
                "WH-001": {"id": "WH-001", "statement": "Old hypothesis"},
            },
        }
        state = ResearchState.from_json(json.dumps(data))
        assert state.hypotheses["WH-001"].depends_on == []


class TestUnestablishedDependencies:
    """Tests for unestablished_dependencies query."""

    def test_returns_working_dependency(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            depends_on=["WH-001"],
        )
        assert state.unestablished_dependencies("WH-002") == ["WH-001"]

    def test_returns_empty_when_dependency_established(self):
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            depends_on=["ER-001"],
        )
        assert state.unestablished_dependencies("WH-002") == []

    def test_returns_empty_for_no_dependencies(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        assert state.unestablished_dependencies("WH-001") == []

    def test_returns_missing_dependency(self):
        """Dependency pointing to non-existent ID is unestablished."""
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            depends_on=["WH-099"],
        )
        assert state.unestablished_dependencies("WH-001") == ["WH-099"]

    def test_returns_empty_for_unknown_hypothesis(self):
        state = ResearchState()
        assert state.unestablished_dependencies("WH-999") == []


# ---------------------------------------------------------------------------
# Phase 4b: Research Questions (B1+B3) tests
# ---------------------------------------------------------------------------


class TestResearchQuestionLifecycle:
    """Tests for ResearchQuestion entity and queries."""

    def test_json_round_trip(self):
        from open_dirac.research_state import ResearchQuestion, RQStatus

        state = ResearchState()
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="What is the leading-order correction?",
            context="Needed for next step",
            resolved_to=["WH-003"],
            status=RQStatus.RESOLVED,
            iteration_created=1,
            iteration_resolved=3,
        )
        json_str = state.to_json()
        restored = ResearchState.from_json(json_str)
        rq = restored.research_questions["RQ-001"]
        assert rq.question == "What is the leading-order correction?"
        assert rq.status == RQStatus.RESOLVED
        assert rq.resolved_to == ["WH-003"]
        assert rq.iteration_resolved == 3

    def test_json_backward_compat_no_rqs(self):
        data = {"hypotheses": {}}
        state = ResearchState.from_json(json.dumps(data))
        assert state.research_questions == {}

    def test_open_research_questions(self):
        from open_dirac.research_state import ResearchQuestion, RQStatus

        state = ResearchState()
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="Open one",
            status=RQStatus.OPEN,
        )
        state.research_questions["RQ-002"] = ResearchQuestion(
            id="RQ-002",
            question="Resolved one",
            status=RQStatus.RESOLVED,
        )
        assert len(state.open_research_questions()) == 1
        assert state.open_research_questions()[0].id == "RQ-001"

    def test_next_rq_num(self):
        from open_dirac.research_state import ResearchQuestion

        state = ResearchState()
        assert state.next_rq_num() == 1
        state.research_questions["RQ-001"] = ResearchQuestion(id="RQ-001")
        assert state.next_rq_num() == 2


# ---------------------------------------------------------------------------
# Evidence on ResearchQuestion
# ---------------------------------------------------------------------------


class TestResearchQuestionEvidence:
    """Evidence can be attached to research questions."""

    def test_rq_evidence_round_trip(self):
        from open_dirac.research_state import ResearchQuestion

        state = ResearchState()
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="What is F?",
            evidence=[
                Evidence(
                    type="research",
                    method="analysis",
                    result="F = pi/4",
                    confidence="exact",
                    iteration=2,
                )
            ],
        )
        restored = ResearchState.from_json(state.to_json())
        rq = restored.research_questions["RQ-001"]
        assert len(rq.evidence) > 0
        assert rq.evidence[0].result == "F = pi/4"
        assert rq.evidence[0].type == "research"

    def test_rq_no_evidence_default(self):
        from open_dirac.research_state import ResearchQuestion

        rq = ResearchQuestion(id="RQ-001", question="Test")
        assert not rq.evidence


# ---------------------------------------------------------------------------
# Evidence on Critique
# ---------------------------------------------------------------------------


class TestCritiqueEvidence:
    """Evidence can be attached to critiques."""

    def test_critique_evidence_round_trip(self):
        from open_dirac.research_state import Critique, Severity, CritiqueStatus

        state = ResearchState()
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.HIGH,
            argument="Spin prediction may be wrong.",
            status=CritiqueStatus.ACTIVE,
            iteration_filed=3,
            evidence=[
                Evidence(
                    type="research",
                    method="re-derivation",
                    result="Spin is indeed 1",
                    confidence="exact",
                    iteration=4,
                )
            ],
        )
        restored = ResearchState.from_json(state.to_json())
        crit = restored.critiques["CRIT-001"]
        assert len(crit.evidence) > 0
        assert crit.evidence[0].result == "Spin is indeed 1"
        assert crit.evidence[0].type == "research"
        assert crit.evidence[0].iteration == 4

    def test_critique_no_evidence_default(self):
        from open_dirac.research_state import Critique

        crit = Critique(id="CRIT-001")
        assert not crit.evidence

    def test_critique_no_evidence_round_trip(self):
        """Critique without evidence survives round-trip (backward compat)."""
        from open_dirac.research_state import Critique, Severity, CritiqueStatus

        state = ResearchState()
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["STRATEGY"],
            severity=Severity.MEDIUM,
            argument="Strategy is vague.",
            status=CritiqueStatus.ACTIVE,
            iteration_filed=2,
        )
        restored = ResearchState.from_json(state.to_json())
        crit = restored.critiques["CRIT-001"]
        assert not crit.evidence
        assert crit.argument == "Strategy is vague."
