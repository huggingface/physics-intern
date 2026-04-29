"""Tests for OrchestratorToolExecutor — state-mutation tools for the orchestrator."""

from unittest.mock import MagicMock

from open_dirac.agents.orchestrator.tools import OrchestratorToolExecutor
from open_dirac.research_state import (
    ResearchState,
    ResearchQuestion,
    Hypothesis,
    HypothesisStatus,
    Verdict,
    Critique,
    Severity,
    CritiqueStatus,
    Evidence,
    ReviewResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace():
    ws = MagicMock()
    ws.root = MagicMock()
    ws.append_file = MagicMock()
    return ws


def _make_state() -> ResearchState:
    """ResearchState with two working hypotheses, no computations or critiques."""
    state = ResearchState()
    state.hypotheses["WH-001"] = Hypothesis(
        id="WH-001",
        statement="First hypothesis",
        status=HypothesisStatus.WORKING,
        derivation="Photon has spin-1.",
        iteration_created=1,
        iteration_modified=1,
    )
    state.hypotheses["WH-002"] = Hypothesis(
        id="WH-002",
        statement="Second hypothesis",
        status=HypothesisStatus.WORKING,
        derivation="Entropy increases in isolated systems.",
        iteration_created=2,
        iteration_modified=2,
    )
    return state


def _make_state_with_verified(target: str = "WH-001") -> ResearchState:
    """State with a VERIFIED review result on *target*."""
    state = _make_state()
    state.hypotheses[target].review = ReviewResult(
        verdict=Verdict.VERIFIED,
        summary=f"Verified {target}",
        iteration=3,
    )
    return state


def _make_state_with_refuted(target: str = "WH-001") -> ResearchState:
    """State with a REFUTED review result on *target*."""
    state = _make_state()
    state.hypotheses[target].review = ReviewResult(
        verdict=Verdict.REFUTED,
        summary=f"Refuted {target}",
        iteration=3,
    )
    return state


def _make_state_with_refuted_and_verified(target: str = "WH-001") -> ResearchState:
    """State with a VERIFIED review result on *target* (supersedes earlier refutation)."""
    state = _make_state()
    state.hypotheses[target].review = ReviewResult(
        verdict=Verdict.VERIFIED,
        summary=f"Verified {target} (corrected)",
        iteration=4,
    )
    return state


def _make_state_with_high_critique(target: str = "WH-001") -> ResearchState:
    """State with VERIFIED review + unresolved HIGH critique targeting *target*."""
    state = _make_state_with_verified(target)
    state.critiques["CRIT-001"] = Critique(
        id="CRIT-001",
        targets=[target],
        severity=Severity.HIGH,
        status=CritiqueStatus.ACTIVE,
        argument="Spin prediction may be wrong.",
    )
    return state


# ---------------------------------------------------------------------------
# add_hypothesis
# ---------------------------------------------------------------------------


class TestAddHypothesis:
    def test_creates_wh003_in_state(self):
        from open_dirac.research_state import ResearchQuestion, RQStatus

        ws = _make_workspace()
        state = _make_state()
        state.hypotheses["WH-002"].status = HypothesisStatus.ESTABLISHED
        state.research_questions["RQ-003"] = ResearchQuestion(
            id="RQ-003",
            question="What is the third result?",
            iteration_created=2,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "add_hypothesis",
            {
                "statement": "Third hypothesis",
                "derivation": "Some derivation.",
                "from_rq": "RQ-003",
            },
        )
        assert not tc.is_error
        assert "WH-003" in tc.output
        assert "WH-003" in state.hypotheses
        h = state.hypotheses["WH-003"]
        assert h.statement == "Third hypothesis"
        assert h.derivation == "Some derivation."
        assert h.status == HypothesisStatus.WORKING
        assert h.iteration_created == 3
        assert h.iteration_modified == 3
        assert ex.mutations_applied
        # RQ should be auto-resolved
        assert state.research_questions["RQ-003"].status == RQStatus.RESOLVED

    def test_rejects_without_from_rq(self):
        ws = _make_workspace()
        state = _make_state()
        state.hypotheses["WH-002"].status = HypothesisStatus.ESTABLISHED
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "add_hypothesis",
            {
                "statement": "Third hypothesis",
                "derivation": "Some derivation.",
            },
        )
        assert "from_rq is required" in tc.output

    def test_blocked_by_wh_cap(self):
        """Cannot create WH when >= 2 working hypotheses exist."""
        from open_dirac.research_state import ResearchQuestion

        ws = _make_workspace()
        state = _make_state()  # 2 working WHs
        state.research_questions["RQ-003"] = ResearchQuestion(
            id="RQ-003",
            question="Q?",
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "add_hypothesis",
            {
                "statement": "Blocked",
                "from_rq": "RQ-003",
            },
        )
        assert "already 2 working hypotheses" in tc.output
        assert "WH-003" not in state.hypotheses

    def test_blocked_by_unresolved_critiques(self):
        """Cannot create WH when unresolved critiques exist."""
        from open_dirac.research_state import (
            Critique,
            CritiqueStatus,
            ResearchQuestion,
            Severity,
        )

        ws = _make_workspace()
        state = _make_state()
        state.hypotheses["WH-002"].status = HypothesisStatus.ESTABLISHED
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.HIGH,
            status=CritiqueStatus.ACTIVE,
            argument="Issue.",
        )
        state.research_questions["RQ-003"] = ResearchQuestion(
            id="RQ-003",
            question="Q?",
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "add_hypothesis",
            {
                "statement": "Blocked",
                "from_rq": "RQ-003",
            },
        )
        assert "blocked" in tc.output
        assert "strategic review" in tc.output


# ---------------------------------------------------------------------------
# RQ evidence cap (dispatch blocked when saturated)
# ---------------------------------------------------------------------------


class TestRqEvidenceCap:
    def _make_rq_with_evidence(self, rq_id: str, n_evidence: int) -> ResearchQuestion:
        from open_dirac.research_state import Evidence

        rq = ResearchQuestion(id=rq_id, question="Q?", iteration_created=1)
        for i in range(n_evidence):
            rq.evidence.append(
                Evidence(id=f"EV-{i + 1:03d}", type="compute", result="r")
            )
        return rq

    def test_dispatch_blocked_when_rq_saturated(self):
        """dispatch_computer is rejected when an open RQ has >= cap evidence."""

        ws = _make_workspace()
        state = ResearchState()
        state.research_questions["RQ-001"] = self._make_rq_with_evidence("RQ-001", 3)
        ex = OrchestratorToolExecutor(
            ws, iteration=5, research_state=state, rq_evidence_cap=3
        )
        tc = ex.execute(
            "dispatch_computer",
            {
                "target_claim": "RQ-001",
                "description": "More work",
            },
        )
        assert "dispatch blocked" in tc.output
        assert "RQ-001" in tc.output
        assert ex.task_data is None  # dispatch did NOT go through

    def test_dispatch_allowed_below_cap(self):
        """dispatch_computer succeeds when RQ evidence count < cap."""
        ws = _make_workspace()
        state = ResearchState()
        state.research_questions["RQ-001"] = self._make_rq_with_evidence("RQ-001", 2)
        ex = OrchestratorToolExecutor(
            ws, iteration=5, research_state=state, rq_evidence_cap=3
        )
        tc = ex.execute(
            "dispatch_computer",
            {
                "target_claim": "RQ-001",
                "description": "More work",
            },
        )
        assert not tc.is_error
        assert ex.task_data is not None

    def test_dispatch_researcher_also_blocked(self):
        """dispatch_researcher is also blocked by saturated RQs."""
        ws = _make_workspace()
        state = ResearchState()
        state.research_questions["RQ-001"] = self._make_rq_with_evidence("RQ-001", 4)
        ex = OrchestratorToolExecutor(
            ws, iteration=5, research_state=state, rq_evidence_cap=3
        )
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "RQ-001",
                "description": "Analyze",
            },
        )
        assert "dispatch blocked" in tc.output

    def test_refuted_evidence_not_counted(self):
        """Refuted evidence does not count toward the cap."""
        from open_dirac.research_state import ResearchQuestion, Evidence

        ws = _make_workspace()
        state = ResearchState()
        rq = ResearchQuestion(id="RQ-001", question="Q?", iteration_created=1)
        for i in range(4):
            rq.evidence.append(
                Evidence(
                    id=f"EV-{i + 1:03d}",
                    type="compute",
                    result="r",
                    refuted=True,
                )
            )
        # One non-refuted — below cap
        rq.evidence.append(Evidence(id="EV-005", type="compute", result="r"))
        state.research_questions["RQ-001"] = rq
        ex = OrchestratorToolExecutor(
            ws, iteration=5, research_state=state, rq_evidence_cap=3
        )
        tc = ex.execute(
            "dispatch_computer",
            {
                "target_claim": "RQ-001",
                "description": "More work",
            },
        )
        assert not tc.is_error


# ---------------------------------------------------------------------------
# abandon_hypothesis
# ---------------------------------------------------------------------------


class TestAbandonHypothesis:
    def test_sets_abandoned_and_creates_failed_approach(self):
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "abandon_hypothesis",
            {
                "id": "WH-001",
                "reason": "Spin prediction was wrong.",
            },
        )
        assert not tc.is_error
        assert state.hypotheses["WH-001"].status == HypothesisStatus.ABANDONED
        assert state.hypotheses["WH-001"].iteration_modified == 3
        assert len(state.failed_approaches) == 1
        fa = state.failed_approaches[0]
        assert "WH-001" in fa.description
        assert "Spin prediction was wrong." in fa.reason
        assert fa.iteration == 3
        assert fa.derivation_excerpt == "Photon has spin-1."
        assert ex.mutations_applied

    def test_abandon_populates_related_entities(self):
        """Abandoning a hypothesis populates related_entities with the hypothesis ID."""
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        ex.execute(
            "abandon_hypothesis",
            {
                "id": "WH-001",
                "reason": "Refuted.",
            },
        )
        fa = state.failed_approaches[0]
        assert fa.related_entities == ["WH-001"]

    def test_abandon_long_derivation_truncated(self):
        """Long derivation is truncated to 300 chars in derivation_excerpt."""
        ws = _make_workspace()
        state = _make_state()
        state.hypotheses["WH-001"].derivation = "A" * 500
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        ex.execute(
            "abandon_hypothesis",
            {
                "id": "WH-001",
                "reason": "Too long.",
            },
        )
        fa = state.failed_approaches[0]
        assert len(fa.derivation_excerpt) == 300

    def test_missing_id_returns_error(self):
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute("abandon_hypothesis", {"id": "WH-099", "reason": "nope"})
        assert "not found" in tc.output

    def test_warns_when_dependents_exist(self):
        ws = _make_workspace()
        state = _make_state()
        # WH-002 depends on WH-001
        state.hypotheses["WH-002"].depends_on = ["WH-001"]
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "abandon_hypothesis",
            {
                "id": "WH-001",
                "reason": "Dead end.",
            },
        )
        assert not tc.is_error
        assert state.hypotheses["WH-001"].status == HypothesisStatus.ABANDONED
        assert "Warning" in tc.output
        assert "WH-002" in tc.output

    def test_no_warning_when_dependent_already_abandoned(self):
        ws = _make_workspace()
        state = _make_state()
        state.hypotheses["WH-002"].depends_on = ["WH-001"]
        state.hypotheses["WH-002"].status = HypothesisStatus.ABANDONED
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "abandon_hypothesis",
            {
                "id": "WH-001",
                "reason": "Dead end.",
            },
        )
        assert not tc.is_error
        assert "Warning" not in tc.output


# ---------------------------------------------------------------------------
# append_convention
# ---------------------------------------------------------------------------


class TestAppendConvention:
    def test_appends_conventions(self):
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "append_convention",
            {
                "content": "- Natural units: h-bar = c = k_B = 1\n- Metric signature: (-,+,+,+)",
            },
        )
        assert not tc.is_error
        assert "Natural units" in state.conventions
        assert ex.mutations_applied

    def test_conventions_append_to_existing(self):
        """Conventions are append-only — new content is appended, not replaced."""
        ws = _make_workspace()
        state = _make_state()
        state.conventions = "Existing conventions."
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "append_convention",
            {
                "content": "New convention added.",
            },
        )
        assert not tc.is_error
        assert "Existing conventions." in state.conventions
        assert "New convention added." in state.conventions

    def test_empty_content_returns_error(self):
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute("append_convention", {"content": "  "})
        assert "empty" in tc.output.lower()


# ---------------------------------------------------------------------------
# append_note
# ---------------------------------------------------------------------------


class TestAppendNote:
    def test_appends_note_to_state(self):
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=4, research_state=state)
        tc = ex.execute(
            "append_note",
            {
                "text": "The sign convention must be checked.",
            },
        )
        assert not tc.is_error
        assert len(state.research_notes) == 1
        assert state.research_notes[0]["text"] == "The sign convention must be checked."
        assert state.research_notes[0]["iteration"] == 4
        assert ex.mutations_applied

    def test_empty_note_returns_error(self):
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=4, research_state=state)
        tc = ex.execute("append_note", {"text": "   "})
        assert "empty" in tc.output.lower()
        assert not ex.mutations_applied

    def test_append_note_no_state_returns_error(self):
        ws = _make_workspace()
        ex = OrchestratorToolExecutor(ws, iteration=4, research_state=None)
        tc = ex.execute("append_note", {"text": "test"})
        assert "no research state" in tc.output


# ---------------------------------------------------------------------------
# Target claim validation — immutable entity guards
# ---------------------------------------------------------------------------


class TestTargetClaimValidation:
    def test_block_dispatch_on_er(self):
        """ERs are immutable — dispatch should be rejected."""
        ws = _make_workspace()
        state = _make_state()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            statement="Established.",
            status=HypothesisStatus.ESTABLISHED,
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {"target_claim": "ER-001", "description": "Re-derive."},
        )
        assert tc.is_error
        assert "immutable" in tc.output.lower() or "Established Results" in tc.output

    def test_block_dispatch_on_resolved_rq(self):
        """Resolved RQs should not receive new evidence."""
        from open_dirac.research_state import ResearchQuestion, RQStatus

        ws = _make_workspace()
        state = _make_state()
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="Test?",
            status=RQStatus.RESOLVED,
            resolved_to=["WH-001"],
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {"target_claim": "RQ-001", "description": "More work."},
        )
        assert tc.is_error
        assert "resolved" in tc.output.lower()

    def test_block_dispatch_on_abandoned_rq(self):
        """Abandoned RQs should not receive new evidence."""
        from open_dirac.research_state import ResearchQuestion, RQStatus

        ws = _make_workspace()
        state = _make_state()
        state.research_questions["RQ-002"] = ResearchQuestion(
            id="RQ-002",
            question="Dead end?",
            status=RQStatus.ABANDONED,
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_computer", {"target_claim": "RQ-002", "description": "Try again."}
        )
        assert tc.is_error
        assert "abandoned" in tc.output.lower()

    def test_allow_dispatch_on_open_rq(self):
        """Open RQs are valid dispatch targets."""
        from open_dirac.research_state import ResearchQuestion

        ws = _make_workspace()
        state = _make_state()
        state.research_questions["RQ-003"] = ResearchQuestion(
            id="RQ-003",
            question="Investigate?",
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher", {"target_claim": "RQ-003", "description": "Derive."}
        )
        assert not tc.is_error

    def test_allow_dispatch_on_working_wh(self):
        """Working WHs (including refuted) are valid dispatch targets."""
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_computer", {"target_claim": "WH-001", "description": "Compute."}
        )
        assert not tc.is_error


# ---------------------------------------------------------------------------
# Dispatch tools
# ---------------------------------------------------------------------------


class TestDispatchTools:
    def test_dispatch_researcher_stores_task_data(self):
        ws = _make_workspace()
        ex = OrchestratorToolExecutor(ws, iteration=3)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "WH-001",
                "description": "Derive result.",
            },
        )
        assert not tc.is_error
        assert ex.task_data["task_type"] == "research"

    def test_dispatch_computer_stores_task_data(self):
        ws = _make_workspace()
        ex = OrchestratorToolExecutor(ws, iteration=3)
        tc = ex.execute(
            "dispatch_computer",
            {
                "target_claim": "WH-001",
                "description": "Compute numerically.",
            },
        )
        assert not tc.is_error
        assert ex.task_data["task_type"] == "compute"

    def test_request_termination_sets_stop(self):
        ws = _make_workspace()
        ex = OrchestratorToolExecutor(ws, iteration=3)
        assert not ex.stop_after_round
        ex.execute(
            "request_termination",
            {
                "reason": "Done.",
                "answer_ers": ["ER-001", "ER-003"],
            },
        )
        assert ex.stop_after_round
        assert ex.task_data["task_type"] == "terminate"
        assert ex.task_data["answer_ers"] == ["ER-001", "ER-003"]

    def test_request_termination_without_reason(self):
        ws = _make_workspace()
        ex = OrchestratorToolExecutor(ws, iteration=3)
        tc = ex.execute("request_termination", {"answer_ers": []})
        assert not tc.is_error
        assert ex.stop_after_round
        assert ex.task_data["description"] == "Research complete."


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


class TestUnknownTool:
    def test_unknown_tool_returns_error(self):
        ws = _make_workspace()
        ex = OrchestratorToolExecutor(ws, iteration=3)
        tc = ex.execute("nonexistent_tool", {})
        assert tc.is_error
        assert "Unknown tool" in tc.output


# ---------------------------------------------------------------------------
# No research_state
# ---------------------------------------------------------------------------


class TestNoResearchState:
    def test_mutation_tools_return_error_without_state(self):
        ws = _make_workspace()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=None)

        mutation_calls = [
            ("add_hypothesis", {"statement": "Test"}),
            ("abandon_hypothesis", {"id": "WH-001", "reason": "test"}),
            ("append_convention", {"content": "test"}),
        ]
        for tool_name, tool_input in mutation_calls:
            tc = ex.execute(tool_name, tool_input)
            assert "no research state" in tc.output, (
                f"{tool_name} should error without research_state"
            )

    def test_dispatch_works_without_state(self):
        ws = _make_workspace()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=None)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "RQ-001",
                "description": "Continue.",
            },
        )
        assert not tc.is_error
        assert ex.task_data is not None
        assert ex.stop_after_round

    def test_request_termination_works_without_state(self):
        ws = _make_workspace()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=None)
        tc = ex.execute("request_termination", {})
        assert not tc.is_error
        assert ex.stop_after_round


class TestDependencyGraph:
    """Tests for depends_on in add_hypothesis."""

    def test_add_hypothesis_with_depends_on(self):
        from open_dirac.research_state import ResearchQuestion

        ws = _make_workspace()
        state = _make_state()
        state.hypotheses["WH-002"].status = HypothesisStatus.ESTABLISHED
        state.research_questions["RQ-003"] = ResearchQuestion(
            id="RQ-003",
            question="Derived from WH-001?",
            iteration_created=2,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "add_hypothesis",
            {
                "statement": "Depends on WH-001",
                "depends_on": ["WH-001"],
                "from_rq": "RQ-003",
            },
        )
        assert not tc.is_error
        new_id = "WH-003"
        assert new_id in state.hypotheses
        assert state.hypotheses[new_id].depends_on == ["WH-001"]


class TestResearchQuestionTools:
    """Tests for add_research_question and abandon_research_question tools."""

    def test_add_research_question(self):
        ws = _make_workspace()
        state = _make_state()  # has WH-001, WH-002
        ex = OrchestratorToolExecutor(ws, iteration=2, research_state=state)
        tc = ex.execute(
            "add_research_question",
            {
                "question": "What is the entropy correction?",
                "context": "Needed for WH-002",
            },
        )
        assert not tc.is_error
        # Shared counter: next number after WH-001/WH-002 is 003
        assert "RQ-003" in tc.output
        assert "RQ-003" in state.research_questions
        rq = state.research_questions["RQ-003"]
        assert rq.question == "What is the entropy correction?"
        assert rq.context == "Needed for WH-002"
        assert ex.mutations_applied

    def test_abandon_research_question(self):
        from open_dirac.research_state import ResearchQuestion, RQStatus

        ws = _make_workspace()
        state = _make_state()
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="What is F(p)?",
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "abandon_research_question",
            {
                "id": "RQ-001",
                "reason": "Dead end after 2 attempts",
            },
        )
        assert not tc.is_error
        rq = state.research_questions["RQ-001"]
        assert rq.status == RQStatus.ABANDONED
        assert rq.iteration_resolved == 3
        assert rq.resolution_reason == "Dead end after 2 attempts"

    def test_abandon_missing_rq_returns_error(self):
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "abandon_research_question",
            {
                "id": "RQ-999",
                "reason": "gone",
            },
        )
        assert "not found" in tc.output

    def test_abandon_already_abandoned_rq_is_idempotent(self):
        """Re-abandoning an already-abandoned RQ returns early without mutation."""
        from open_dirac.research_state import ResearchQuestion, RQStatus

        ws = _make_workspace()
        state = _make_state()
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="What is F(p)?",
            status=RQStatus.ABANDONED,
            iteration_created=1,
            iteration_resolved=2,
            resolution_reason="Dead end",
        )
        ex = OrchestratorToolExecutor(ws, iteration=5, research_state=state)
        tc = ex.execute(
            "abandon_research_question",
            {
                "id": "RQ-001",
                "reason": "Trying to abandon again",
            },
        )
        assert "already abandoned" in tc.output
        rq = state.research_questions["RQ-001"]
        assert rq.iteration_resolved == 2
        assert rq.resolution_reason == "Dead end"
        assert not ex.mutations_applied

    def test_abandon_resolved_rq_returns_error(self):
        """Cannot abandon an RQ that was auto-resolved by add_hypothesis."""
        from open_dirac.research_state import ResearchQuestion, RQStatus

        ws = _make_workspace()
        state = _make_state()
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="What is F(p)?",
            status=RQStatus.RESOLVED,
            iteration_created=1,
            iteration_resolved=2,
            resolution_reason="Promoted to WH-001",
        )
        ex = OrchestratorToolExecutor(ws, iteration=5, research_state=state)
        tc = ex.execute(
            "abandon_research_question",
            {
                "id": "RQ-001",
                "reason": "Trying to abandon a resolved RQ",
            },
        )
        assert "already resolved" in tc.output
        assert not ex.mutations_applied

    def test_add_rq_blocked_by_cap(self):
        """Cannot create RQ when >= max_open_rqs open RQs exist."""
        from open_dirac.research_state import ResearchQuestion

        ws = _make_workspace()
        state = _make_state()
        for i in range(3, 6):
            rq_id = f"RQ-{i:03d}"
            state.research_questions[rq_id] = ResearchQuestion(
                id=rq_id,
                question=f"Q{i}?",
                iteration_created=1,
            )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute("add_research_question", {"question": "One more?"})
        assert "already 3 open RQ(s)" in tc.output
        assert "limit is 1" in tc.output

    def test_add_rq_blocked_by_unresolved_critiques(self):
        """Cannot create RQ when unresolved critiques exist."""
        from open_dirac.research_state import Critique, CritiqueStatus, Severity

        ws = _make_workspace()
        state = _make_state()
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.HIGH,
            status=CritiqueStatus.ACTIVE,
            argument="Issue.",
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute("add_research_question", {"question": "Blocked?"})
        assert "blocked" in tc.output
        assert "strategic review" in tc.output

    def test_add_hypothesis_not_blocked_by_medium_critique(self):
        """MEDIUM critique does not block WH creation (severity-gated)."""
        from open_dirac.research_state import (
            Critique,
            CritiqueStatus,
            ResearchQuestion,
            Severity,
        )

        ws = _make_workspace()
        state = _make_state()
        state.hypotheses["WH-002"].status = HypothesisStatus.ESTABLISHED
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.MEDIUM,
            status=CritiqueStatus.ACTIVE,
            argument="Minor concern.",
        )
        state.research_questions["RQ-003"] = ResearchQuestion(
            id="RQ-003",
            question="Q?",
            iteration_created=1,
            evidence=[Evidence(id="EV-001", type="research", result="r")],
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "add_hypothesis",
            {
                "statement": "Allowed",
                "from_rq": "RQ-003",
            },
        )
        assert "Error" not in tc.output or "critique" not in tc.output

    def test_add_rq_not_blocked_by_low_critique(self):
        """LOW critique does not block RQ creation (severity-gated)."""
        from open_dirac.research_state import Critique, CritiqueStatus, Severity

        ws = _make_workspace()
        state = ResearchState()
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["STRATEGY"],
            severity=Severity.LOW,
            status=CritiqueStatus.ACTIVE,
            argument="Cosmetic issue.",
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute("add_research_question", {"question": "Allowed?"})
        assert "critique" not in tc.output.lower()

    def test_add_hypothesis_from_already_resolved_rq_blocked(self):
        """Creating a WH from an already-resolved RQ is rejected."""
        from open_dirac.research_state import ResearchQuestion, RQStatus

        ws = _make_workspace()
        state = _make_state()
        state.hypotheses["WH-002"].status = HypothesisStatus.ESTABLISHED
        state.research_questions["RQ-003"] = ResearchQuestion(
            id="RQ-003",
            question="What is the entropy?",
            status=RQStatus.RESOLVED,
            iteration_created=1,
            iteration_resolved=2,
            resolution_reason="Answered during exploration",
        )
        ex = OrchestratorToolExecutor(ws, iteration=5, research_state=state)
        tc = ex.execute(
            "add_hypothesis",
            {
                "statement": "S = 4 pi M^2",
                "from_rq": "RQ-003",
            },
        )
        assert "already resolved" in tc.output
        # No WH created
        assert "WH-003" not in state.hypotheses


# ---------------------------------------------------------------------------
# Dispatch gate
# ---------------------------------------------------------------------------


class TestDispatchGate:
    """Tests for dispatch tool behavior with mutations."""

    @staticmethod
    def _state_with_open_rq():
        """State with WH-001 (working), WH-002 (established), and an open RQ-003 for add_hypothesis calls."""
        from open_dirac.research_state import ResearchQuestion

        state = _make_state()
        state.hypotheses["WH-002"].status = HypothesisStatus.ESTABLISHED
        state.research_questions["RQ-003"] = ResearchQuestion(
            id="RQ-003",
            question="Placeholder for dispatch test",
            iteration_created=2,
        )
        return state

    def test_add_hypothesis_acts_as_dispatch(self):
        """add_hypothesis sets stop_after_round and task_data for auto-review."""
        ws = _make_workspace()
        state = self._state_with_open_rq()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)

        tc = ex.execute(
            "add_hypothesis", {"statement": "New claim", "from_rq": "RQ-003"}
        )
        assert not tc.is_error
        assert "WH-003" in state.hypotheses
        assert ex.stop_after_round
        assert ex.task_data is not None
        assert ex.task_data["task_type"] == "review"
        assert ex.task_data["target_claim"] == "WH-003"
        assert "Review will be dispatched automatically" in tc.output

    def test_multiple_entity_mutations_in_same_round(self):
        """Multiple entity-creating mutations in the same response all execute."""
        ws = _make_workspace()
        state = self._state_with_open_rq()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)

        tc1 = ex.execute("add_hypothesis", {"statement": "First", "from_rq": "RQ-003"})
        assert not tc1.is_error
        tc2 = ex.execute("add_research_question", {"question": "Question?"})
        assert not tc2.is_error
        assert "WH-003" in state.hypotheses
        assert "RQ-004" in state.research_questions

    def test_dispatch_alone_succeeds(self):
        """dispatch_researcher succeeds when no mutations occurred this round."""
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)

        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "WH-001",
                "description": "Derive result.",
            },
        )
        assert "Dispatched" in tc.output
        assert ex.task_data is not None
        assert ex.stop_after_round

    def test_dispatch_works_across_rounds(self):
        """Mutation in round 1 → begin_round → dispatch in round 2 succeeds."""
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)

        # Round 1: mutation only
        ex.execute("append_convention", {"content": "Natural units."})

        # Round 2: dispatch
        ex.begin_round()
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "WH-001",
                "description": "Derive result.",
            },
        )
        assert "Dispatched" in tc.output
        assert ex.stop_after_round
        assert ex.task_data["target_claim"] == "WH-001"

    def test_tools_in_prior_round_dont_block_dispatch(self):
        """Tool calls in an earlier response don't affect dispatch in a later one."""
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)

        # Round 1: mutations only
        ex.execute("append_convention", {"content": "Natural units."})
        assert ex._calls_this_round == 1

        # Round 2: new response, dispatch only
        ex.begin_round()
        assert ex._calls_this_round == 0
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "WH-001",
                "description": "Derive result.",
            },
        )
        assert "Dispatched" in tc.output
        assert ex.stop_after_round

    def test_non_entity_mutations_plus_dispatch_succeed(self):
        """update + append_convention + dispatch all succeed."""
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)

        ex.execute("append_convention", {"content": "Natural units."})
        ex.execute(
            "append_convention",
            {
                "content": "Natural units.",
            },
        )
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "WH-001",
                "description": "Derive result.",
            },
        )
        assert "Dispatched" in tc.output
        assert ex.stop_after_round
        assert ex.task_data is not None

    def test_mutations_plus_dispatch_succeed(self):
        """append_note + dispatch in same round all succeed."""
        ws = _make_workspace()
        state = _make_state_with_verified("WH-001")
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)

        ex.execute("append_note", {"text": "Test note"})
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "WH-002",
                "description": "Derive result.",
            },
        )
        assert "Dispatched" in tc.output
        assert ex.stop_after_round


# ---------------------------------------------------------------------------
# State injection
# ---------------------------------------------------------------------------


class TestStateInjection:
    """Tests for the state injection rendered by end_round()."""

    def test_render_state_injection_after_mutations(self):
        """Injection includes applied mutations and state snapshot."""
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)

        # Round 1: mutation
        ex.execute("append_note", {"text": "Test observation"})

        # end_round returns injection for THIS round's mutations
        injection = ex.end_round()
        assert injection is not None
        assert "Appended research note" in injection
        assert "State snapshot" in injection

    def test_render_state_injection_none_without_mutations(self):
        """No injection when no mutations occurred."""
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)

        injection = ex.end_round()
        assert injection is None

    def test_injection_shows_unreviewed_wh(self):
        """WH with evidence + no review → guidance line in injection."""
        ws = _make_workspace()
        state = _make_state()
        state.hypotheses["WH-001"].evidence = [
            Evidence(
                type="research",
                summary="Some evidence",
                iteration=2,
            )
        ]
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)

        # Trigger a mutation so injection fires
        ex.execute("append_note", {"text": "Test note"})
        injection = ex.end_round()
        assert injection is not None
        assert "WH-001 awaiting auto-review" in injection

    def test_injection_shows_verified_pending_guidance(self):
        """WH with VERIFIED review but unestablished deps → pending guidance."""
        ws = _make_workspace()
        state = _make_state_with_verified("WH-001")
        state.hypotheses["WH-001"].depends_on = ["WH-002"]
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)

        ex.execute("append_note", {"text": "Test note"})
        injection = ex.end_round()
        assert injection is not None
        assert "WH-001 is VERIFIED, pending auto-promotion" in injection
        assert "WH-002" in injection

    def test_injection_shows_budget_pressure(self):
        """Low iterations remaining → budget message in injection."""
        ws = _make_workspace()
        state = ResearchState()
        # Set up ER directly
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            statement="Second",
            status=HypothesisStatus.WORKING,
            iteration_created=2,
            iteration_modified=2,
        )
        ex = OrchestratorToolExecutor(
            ws,
            iteration=18,
            research_state=state,
            max_iterations=20,
            budget_synthesis_margin=3,
        )

        ex.execute("append_note", {"text": "Budget check"})
        injection = ex.end_round()
        assert injection is not None
        assert "BUDGET" in injection
        assert "2 iteration(s) remaining" in injection

    def test_injection_shows_completion_ready(self):
        """All resolved → terminate guidance in injection."""
        ws = _make_workspace()
        state = ResearchState()
        for i in range(1, 4):
            state.hypotheses[f"ER-{i:03d}"] = Hypothesis(
                id=f"ER-{i:03d}",
                status=HypothesisStatus.ESTABLISHED,
            )
        ex = OrchestratorToolExecutor(
            ws,
            iteration=5,
            research_state=state,
            min_er_for_completion=3,
        )

        ex.execute("append_note", {"text": "Final check"})
        injection = ex.end_round()
        assert injection is not None
        assert "All entities resolved" in injection
        assert "terminate" in injection.lower()

    def test_simplified_tool_results_no_nudge(self):
        """Tool results don't contain old _BATCH_NUDGE text."""
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)

        nudge = "Call a dispatch tool"
        tc1 = ex.execute("add_hypothesis", {"statement": "Test"})
        assert nudge not in tc1.output
        tc2 = ex.execute("append_convention", {"content": "Natural units."})
        assert nudge not in tc2.output
        tc3 = ex.execute("append_note", {"text": "Note"})
        assert nudge not in tc3.output


# ---------------------------------------------------------------------------
# Target claim validation
# ---------------------------------------------------------------------------


class TestTargetClaimValidationSuccess:
    """Tests for valid target_claim values in dispatch tools."""

    def test_valid_wh_target_passes(self):
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "WH-001",
                "description": "Derive result.",
            },
        )
        assert "Dispatched" in tc.output
        assert ex.stop_after_round

    def test_valid_rq_target_passes(self):
        from open_dirac.research_state import ResearchQuestion

        ws = _make_workspace()
        state = _make_state()
        state.research_questions["RQ-003"] = ResearchQuestion(
            id="RQ-003",
            question="Test?",
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "RQ-003",
                "description": "Explore.",
            },
        )
        assert "Dispatched" in tc.output
        assert ex.stop_after_round

    def test_invalid_target_rejected_with_listing(self):
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "WH-099",
                "description": "Verify.",
            },
        )
        assert "Error" in tc.output
        assert "WH-099" in tc.output
        assert "WH-001" in tc.output  # listed as valid
        assert "WH-002" in tc.output
        assert ex.task_data is None
        assert not ex.stop_after_round

    def test_termination_skips_validation(self):
        ws = _make_workspace()
        state = _make_state()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute("request_termination", {"reason": "Done."})
        assert "Termination" in tc.output
        assert ex.stop_after_round

    def test_crit_target_passes_as_unknown_prefix(self):
        """CRIT prefix is no longer recognized — passes through as unknown prefix."""
        ws = _make_workspace()
        state = _make_state()
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.HIGH,
            status=CritiqueStatus.ACTIVE,
            argument="Spin prediction may be wrong.",
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "CRIT-001",
                "description": "Investigate critique.",
            },
        )
        assert "Dispatched" in tc.output
        assert ex.stop_after_round

    def test_crit_target_treated_as_unknown_prefix(self):
        """CRIT targets are no longer recognized — they pass as unknown prefix."""
        ws = _make_workspace()
        state = _make_state()
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.HIGH,
            status=CritiqueStatus.ACTIVE,
            argument="Spin prediction may be wrong.",
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "CRIT-099",
                "description": "Investigate critique.",
            },
        )
        # CRIT prefix is no longer in the regex — passes as unknown prefix
        assert "Dispatched" in tc.output
        assert ex.stop_after_round

    def test_skipped_when_research_state_is_none(self):
        ws = _make_workspace()
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=None)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "WH-099",
                "description": "Derive result.",
            },
        )
        assert "Dispatched" in tc.output
        assert ex.stop_after_round


# ---------------------------------------------------------------------------
# Focus guard (_check_focus)
# ---------------------------------------------------------------------------


class TestFocusGuard:
    """Tests for serial RQ focus, dangling WH blocking, and refuted retry cap."""

    def test_dangling_refuted_wh_blocks_rq_dispatch(self):
        """A REFUTED WH blocks dispatch to any RQ."""
        from open_dirac.research_state import ResearchQuestion

        ws = _make_workspace()
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="Claim",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict="REFUTED", summary="Wrong", iteration=2),
            refuted_count=1,
        )
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="Q?",
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "RQ-001",
                "description": "Investigate.",
            },
        )
        assert "Error" in tc.output
        assert "WH-001" in tc.output
        assert "unresolved WH" in tc.output
        assert not ex.stop_after_round

    def test_dangling_inconclusive_wh_blocks_rq_dispatch(self):
        """An INCONCLUSIVE WH blocks dispatch to any RQ."""
        from open_dirac.research_state import ResearchQuestion

        ws = _make_workspace()
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="Claim",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict="INCONCLUSIVE", summary="Unclear", iteration=2),
        )
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="Q?",
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_computer",
            {
                "target_claim": "RQ-001",
                "description": "Compute.",
            },
        )
        assert "Error" in tc.output
        assert "WH-001" in tc.output
        assert not ex.stop_after_round

    def test_dangling_wh_allows_wh_dispatch(self):
        """A dangling WH does NOT block dispatch to that WH itself."""
        ws = _make_workspace()
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="Claim",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict="REFUTED", summary="Wrong", iteration=2),
            refuted_count=1,
        )
        ex = OrchestratorToolExecutor(
            ws, iteration=3, research_state=state, max_refuted_retries=3
        )
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "WH-001",
                "description": "Retry.",
            },
        )
        assert "Dispatched" in tc.output
        assert ex.stop_after_round

    def test_serial_rq_blocks_second_rq_with_evidence(self):
        """Cannot dispatch to RQ-002 when RQ-001 already has evidence."""
        from open_dirac.research_state import ResearchQuestion

        ws = _make_workspace()
        state = ResearchState()
        rq1 = ResearchQuestion(id="RQ-001", question="Q1?", iteration_created=1)
        rq1.evidence.append(Evidence(type="compute", result="r"))
        state.research_questions["RQ-001"] = rq1
        state.research_questions["RQ-002"] = ResearchQuestion(
            id="RQ-002",
            question="Q2?",
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "RQ-002",
                "description": "Investigate.",
            },
        )
        assert "Error" in tc.output
        assert "serial RQ focus" in tc.output
        assert "RQ-001" in tc.output
        assert not ex.stop_after_round

    def test_serial_rq_allows_same_rq(self):
        """Dispatch to RQ-001 allowed when RQ-001 already has evidence."""
        from open_dirac.research_state import ResearchQuestion

        ws = _make_workspace()
        state = ResearchState()
        rq1 = ResearchQuestion(id="RQ-001", question="Q1?", iteration_created=1)
        rq1.evidence.append(Evidence(type="compute", result="r"))
        state.research_questions["RQ-001"] = rq1
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "RQ-001",
                "description": "More evidence.",
            },
        )
        assert "Dispatched" in tc.output
        assert ex.stop_after_round

    def test_serial_rq_allows_when_no_evidence(self):
        """Two open RQs with no evidence — dispatch to either is fine."""
        from open_dirac.research_state import ResearchQuestion

        ws = _make_workspace()
        state = ResearchState()
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="Q1?",
            iteration_created=1,
        )
        state.research_questions["RQ-002"] = ResearchQuestion(
            id="RQ-002",
            question="Q2?",
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "RQ-002",
                "description": "Investigate.",
            },
        )
        assert "Dispatched" in tc.output

    def test_refuted_retry_cap_blocks_dispatch(self):
        """WH at refuted_count >= max_refuted_retries blocks dispatch."""
        ws = _make_workspace()
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="Claim",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict="REFUTED", summary="Wrong", iteration=2),
            refuted_count=2,
        )
        ex = OrchestratorToolExecutor(
            ws, iteration=3, research_state=state, max_refuted_retries=2
        )
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "WH-001",
                "description": "Retry.",
            },
        )
        assert "Error" in tc.output
        assert "refuted" in tc.output
        assert "abandon" in tc.output.lower()
        assert not ex.stop_after_round

    def test_refuted_retry_cap_allows_below_limit(self):
        """WH with refuted_count < max_refuted_retries allows dispatch."""
        ws = _make_workspace()
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="Claim",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict="REFUTED", summary="Wrong", iteration=2),
            refuted_count=1,
        )
        ex = OrchestratorToolExecutor(
            ws, iteration=3, research_state=state, max_refuted_retries=2
        )
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "WH-001",
                "description": "Retry.",
            },
        )
        assert "Dispatched" in tc.output
        assert ex.stop_after_round

    def test_verified_wh_does_not_block_rq_dispatch(self):
        """VERIFIED WHs are not dangling — don't block RQ dispatch."""
        from open_dirac.research_state import ResearchQuestion

        ws = _make_workspace()
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="Claim",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict="VERIFIED", summary="Good", iteration=2),
        )
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="Q?",
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "RQ-001",
                "description": "Investigate.",
            },
        )
        assert "Dispatched" in tc.output

    def test_no_review_wh_does_not_block_rq_dispatch(self):
        """WH without any review is not dangling — doesn't block RQ dispatch."""
        from open_dirac.research_state import ResearchQuestion

        ws = _make_workspace()
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="Claim",
            status=HypothesisStatus.WORKING,
        )
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="Q?",
            iteration_created=1,
        )
        ex = OrchestratorToolExecutor(ws, iteration=3, research_state=state)
        tc = ex.execute(
            "dispatch_researcher",
            {
                "target_claim": "RQ-001",
                "description": "Investigate.",
            },
        )
        assert "Dispatched" in tc.output
