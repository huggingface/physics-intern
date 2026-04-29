"""Tests for OpenDirac engine (research status, budget enforcement, overrides)."""

from unittest.mock import MagicMock, patch

from open_dirac.config import Config
from open_dirac.loop_state import DispatchRecord, LoopState
from open_dirac.research_state import Evidence, Hypothesis, ResearchState, ReviewResult
from open_dirac.task import Task, TaskType
from open_dirac.validation import Violation, ViolationSeverity


def _safe_path_mock() -> MagicMock:
    """MagicMock safe for ``ws.root.__truediv__``.

    A bare ``MagicMock`` has ``__int__`` → 1, so ``open(mock)`` silently
    opens fd 1 (stdout) and the ``with`` block closes it.  This helper
    returns a mock whose ``return_value.exists()`` → ``False``, preventing
    the engine's ``_run_formal_verification`` from reaching ``open()``.
    """
    m = MagicMock()
    m.return_value.exists.return_value = False
    return m


class TestSetResearchStatus:
    """Test _set_research_status updates research state."""

    def test_set_research_status(self):
        from open_dirac.engine import OpenDirac

        engine = OpenDirac.__new__(OpenDirac)
        engine.config = Config()
        engine.research_state = ResearchState()

        engine._set_research_status("completed")

        assert engine.research_state.status == "completed"


class TestEnrichComputeTask:
    """Test compute task enrichment with prior failure context."""

    def _make_engine(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"
            written = {}

            def capture_write(filename, content):
                written[filename] = content

            ws.write_file = MagicMock(side_effect=capture_write)

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.research_state = ResearchState()
            # Record prior failures via claim_failure_count in LoopState
            engine._state = LoopState()
            engine._state.claim_failure_count["WH-003"] = 2
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 3
        return engine, ws, written

    def test_enrich_compute_task_appends_context(self):
        """Prior failures exist -> CURRENT_TASK enriched."""
        engine, ws, written = self._make_engine()
        ws.read_file = MagicMock(
            return_value="---\ntask_type: compute\n---\n\nCompute WH-003 mass."
        )

        task = Task(
            task_id="TASK-003",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            body="Compute WH-003 mass limit",
        )
        engine._enrich_compute_task_with_prior_failures(task)

        assert "CURRENT_TASK.md" in written
        enriched = written["CURRENT_TASK.md"]
        assert "Prior Failure Context" in enriched
        assert "2 prior failure(s)" in enriched
        assert "ROOT CAUSE" in enriched

    def test_enrich_compute_task_no_match(self):
        """No prior failures on this target -> unchanged."""
        engine, ws, written = self._make_engine()
        ws.read_file = MagicMock(
            return_value="---\ntask_type: compute\n---\n\nCompute WH-099."
        )

        task = Task(
            task_id="TASK-003",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            body="Compute WH-099 something new",
        )
        engine._enrich_compute_task_with_prior_failures(task)

        assert "CURRENT_TASK.md" not in written  # write_file not called


class TestComputeVerdictTracking:
    """Test dispatch-level verdict tracking with failure counter and verdict signals."""

    def _make_engine(self, stall_recompute_limit=2):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config(stall_recompute_limit=stall_recompute_limit)
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 3
            engine._state = LoopState()
            engine.research_state = ResearchState()
        return engine

    def _set_verification(self, engine, target, verdict_str, summary=""):
        from open_dirac.research_state import Hypothesis, ReviewResult

        if target not in engine.research_state.hypotheses:
            engine.research_state.hypotheses[target] = Hypothesis(id=target)
        engine.research_state.hypotheses[target].review = ReviewResult(
            verdict=verdict_str,
            summary=summary,
            iteration=engine.iteration,
        )

    def test_refuted_signals_orchestrator(self):
        """REFUTED verdict adds to pending_compute_verdicts."""
        engine = self._make_engine()
        self._set_verification(engine, "WH-001", "REFUTED")
        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Verify formula X = Y",
        )
        engine._track_agent_result(task)

        assert len(engine._state.pending_compute_verdicts) == 1
        assert engine._state.pending_compute_verdicts[0]["verdict"] == "REFUTED"
        assert engine._state.pending_compute_verdicts[0]["attempt"] == 1

    def test_inconclusive_signals_orchestrator(self):
        """INCONCLUSIVE also counted and signals orchestrator."""
        engine = self._make_engine()
        self._set_verification(engine, "WH-001", "INCONCLUSIVE")
        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Verify formula X = Y",
        )
        engine._track_agent_result(task)

        assert len(engine._state.pending_compute_verdicts) == 1
        assert engine._state.pending_compute_verdicts[0]["verdict"] == "INCONCLUSIVE"

    def test_refuted_marks_evidence_as_refuted(self):
        """REFUTED verdict marks all existing evidence on hypothesis as refuted."""
        engine = self._make_engine()
        h = Hypothesis(
            id="WH-001",
            evidence=[
                Evidence(type="compute", result="wrong answer", iteration=1),
                Evidence(type="research", result="also wrong", iteration=2),
            ],
        )
        engine.research_state.hypotheses["WH-001"] = h
        h.review = ReviewResult(verdict="REFUTED", summary="bad", iteration=3)

        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Review WH-001",
        )
        engine._track_agent_result(task)

        assert all(ev.refuted for ev in h.evidence)

    def test_refuted_keeps_working_and_increments_refuted_count(self):
        """REFUTED verdict keeps WH WORKING, increments refuted_count, no FailedApproach."""
        from open_dirac.research_state import HypothesisStatus

        engine = self._make_engine()
        h = Hypothesis(
            id="WH-001",
            statement="Claim X = Y",
            evidence=[
                Evidence(type="compute", result="wrong", iteration=1),
            ],
        )
        engine.research_state.hypotheses["WH-001"] = h
        h.review = ReviewResult(
            verdict="REFUTED", summary="Derivation has an error in step 3", iteration=3
        )

        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Review WH-001",
        )
        engine._track_agent_result(task)

        assert h.status == HypothesisStatus.WORKING
        assert h.refuted_count == 1
        assert len(engine.research_state.failed_approaches) == 0
        # Evidence is still marked as refuted
        assert all(ev.refuted for ev in h.evidence)

    def test_inconclusive_keeps_evidence(self):
        """INCONCLUSIVE verdict keeps existing evidence (not wrong, just insufficient)."""
        engine = self._make_engine()
        h = Hypothesis(
            id="WH-001",
            evidence=[
                Evidence(type="compute", result="unclear", iteration=1),
            ],
        )
        engine.research_state.hypotheses["WH-001"] = h
        h.review = ReviewResult(verdict="INCONCLUSIVE", summary="unclear", iteration=2)

        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Review WH-001",
        )
        engine._track_agent_result(task)

        assert not any(ev.refuted for ev in h.evidence)

    def test_verified_does_not_mark_evidence_refuted(self):
        """VERIFIED verdict leaves evidence untouched."""
        engine = self._make_engine()
        h = Hypothesis(
            id="WH-001",
            evidence=[
                Evidence(type="compute", result="correct", iteration=1),
            ],
        )
        engine.research_state.hypotheses["WH-001"] = h
        h.review = ReviewResult(verdict="VERIFIED", summary="good", iteration=2)

        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Review WH-001",
        )
        engine._track_agent_result(task)

        assert not any(ev.refuted for ev in h.evidence)

    def test_stalled_verdict_signal(self):
        """After N failures, signal says STALLED in context suffix."""
        engine = self._make_engine()
        self._set_verification(engine, "WH-001", "REFUTED")
        engine._state.claim_failure_count["WH-001"] = 1  # next will be 2 == limit

        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Verify formula X = Y",
        )
        engine._track_agent_result(task)

        assert len(engine._state.pending_compute_verdicts) == 1
        assert engine._state.pending_compute_verdicts[0]["attempt"] == 2
        prefix = engine._build_context_suffix()
        assert "STALLED" in prefix
        assert "do NOT schedule another review" in prefix

    def test_verified_clears_failure_count(self):
        """VERIFIED clears the failure counter and populates pending_verified_results."""
        engine = self._make_engine()
        self._set_verification(engine, "WH-001", "VERIFIED")
        engine._state.claim_failure_count["WH-001"] = 1

        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Verify formula X = Y",
        )
        engine._track_agent_result(task)

        assert "WH-001" not in engine._state.claim_failure_count
        assert len(engine._state.pending_compute_verdicts) == 0
        assert len(engine._state.pending_verified_results) == 1

    def test_verified_populates_dict_with_correct_keys(self):
        """VERIFIED verification populates dict with claim and verdict."""
        engine = self._make_engine()
        self._set_verification(engine, "WH-001", "VERIFIED")
        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Verify formula X = Y",
        )
        engine._track_agent_result(task)

        assert len(engine._state.pending_verified_results) == 1
        v = engine._state.pending_verified_results[0]
        assert v["claim"] == "ER-001"  # updated to ER after auto-promote
        assert v["verdict"] == "VERIFIED"

    def test_verified_banner_renders_and_consumed_once(self):
        """VERIFIED HYPOTHESES banner renders in context suffix and is consumed."""
        engine = self._make_engine()
        engine._state.pending_verified_results = [
            {
                "claim": "WH-001",
                "verdict": "VERIFIED",
            }
        ]
        prefix = engine._build_context_suffix()
        assert "VERIFIED HYPOTHESES" in prefix
        assert "WH-001 VERIFIED by reviewer" in prefix
        # Consumed
        assert len(engine._state.pending_verified_results) == 0
        # Second call should be empty
        prefix2 = engine._build_context_suffix()
        assert "VERIFIED HYPOTHESES" not in prefix2

    def test_verified_banner_with_claim_id(self):
        """VERIFIED banner includes claim ID."""
        engine = self._make_engine()
        engine._state.pending_verified_results = [
            {
                "claim": "WH-001",
                "verdict": "VERIFIED",
            }
        ]
        prefix = engine._build_context_suffix()
        assert "WH-001" in prefix

    def test_verified_banner_ordering(self):
        """VERIFIED banner appears after evidence results, before verification results."""
        engine = self._make_engine()
        engine._state.pending_explore_results = [
            {
                "target_id": "WH-002",
                "description": "Explore result",
                "result": "x = 42",
                "confidence": "exact",
                "task_id": "TASK-001",
                "task_type": "compute",
            }
        ]
        engine._state.pending_verified_results = [
            {
                "claim": "WH-001",
                "verdict": "VERIFIED",
            }
        ]
        engine._state.pending_compute_verdicts = [
            {
                "verdict": "REFUTED",
                "claim": "WH-003",
                "attempt": 1,
                "notes": "",
            }
        ]
        prefix = engine._build_context_suffix()
        explore_pos = prefix.index("EVIDENCE RESULTS")
        verified_pos = prefix.index("VERIFIED HYPOTHESES")
        verdicts_pos = prefix.index("VERIFICATION RESULTS")
        assert explore_pos < verified_pos < verdicts_pos

    def test_provenance_in_evidence_banner(self):
        """Evidence results banner includes task provenance."""
        engine = self._make_engine()
        engine._state.pending_explore_results = [
            {
                "target_id": "RQ-001",
                "description": "Derived formula",
                "result": "T = 1/(8*pi*M)",
                "confidence": "exact",
                "task_id": "TASK-002",
                "task_type": "research",
            }
        ]
        suffix = engine._build_context_suffix()
        assert "[from TASK-002: research on RQ-001]" in suffix

    def test_failed_evidence_no_consider_hint(self):
        """Failed evidence shows NOTE instead of Consider hint."""
        engine = self._make_engine()
        engine._state.pending_explore_results = [
            {
                "target_id": "RQ-001",
                "description": "unknown",
                "result": "Agent produced no exit tool call.",
                "confidence": "partial",
                "task_id": "TASK-001",
                "task_type": "compute",
            }
        ]
        suffix = engine._build_context_suffix()
        assert "do NOT treat it as usable evidence" in suffix
        assert "Consider: formulate" not in suffix

    def test_failed_research_parse_no_consider_hint(self):
        """Failed research parse shows NOTE instead of Consider hint."""
        engine = self._make_engine()
        engine._state.pending_explore_results = [
            {
                "target_id": "RQ-002",
                "description": "unknown",
                "result": "Failed to parse structured research output.",
                "confidence": "partial",
                "task_id": "TASK-002",
                "task_type": "research",
            }
        ]
        suffix = engine._build_context_suffix()
        assert "do NOT treat it as usable evidence" in suffix
        assert "Consider: formulate" not in suffix

    def test_successful_evidence_has_consider_hint(self):
        """Successful evidence retains the Consider hint."""
        engine = self._make_engine()
        engine._state.pending_explore_results = [
            {
                "target_id": "RQ-003",
                "description": "Computed value",
                "result": "T = 1/(8*pi*M)",
                "confidence": "exact",
                "task_id": "TASK-003",
                "task_type": "compute",
            }
        ]
        suffix = engine._build_context_suffix()
        assert "do NOT treat it as usable evidence" not in suffix

    def test_provenance_in_verified_banner(self):
        """Verified hypotheses banner includes task provenance."""
        engine = self._make_engine()
        engine._state.pending_verified_results = [
            {
                "claim": "WH-001",
                "verdict": "VERIFIED",
                "task_id": "TASK-003",
            }
        ]
        suffix = engine._build_context_suffix()
        assert "[from TASK-003]" in suffix

    def test_provenance_in_verdict_banner(self):
        """Verification results banner includes task provenance."""
        engine = self._make_engine()
        engine._state.pending_compute_verdicts = [
            {
                "verdict": "REFUTED",
                "claim": "WH-001",
                "attempt": 1,
                "notes": "",
                "task_id": "TASK-004",
            }
        ]
        suffix = engine._build_context_suffix()
        assert "[from TASK-004]" in suffix

    def test_track_agent_result_stores_provenance(self):
        """_track_agent_result stores task_id and task_type in pending results."""
        engine = self._make_engine()
        from open_dirac.research_state import Evidence

        rq = engine.research_state.research_questions.get("RQ-001")
        if rq is None:
            from open_dirac.research_state import ResearchQuestion

            engine.research_state.research_questions["RQ-001"] = ResearchQuestion(
                id="RQ-001",
                question="What is T?",
            )
        engine.research_state.research_questions["RQ-001"].evidence = [
            Evidence(
                type="research",
                reasoning="Derived",
                method="algebra",
                result="T = 1/(8*pi*M)",
                confidence="exact",
                iteration=1,
            )
        ]
        task = Task(
            task_id="TASK-005",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            target_claim="RQ-001",
            body="Derive temperature",
        )
        engine._track_agent_result(task)

        assert len(engine._state.pending_explore_results) == 1
        r = engine._state.pending_explore_results[0]
        assert r["task_id"] == "TASK-005"
        assert r["task_type"] == "research"

    def test_different_claims_tracked_independently(self):
        """Two different WH IDs have separate counters."""
        engine = self._make_engine()
        self._set_verification(engine, "WH-002", "REFUTED")
        engine._state.claim_failure_count["WH-001"] = 1

        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-002",
            body="Verify WH-002 temperature",
        )
        engine._track_agent_result(task)

        assert engine._state.claim_failure_count["WH-001"] == 1
        assert engine._state.claim_failure_count.get("WH-002", 0) == 1

    def test_refuted_with_notes_populates_dict(self):
        """REFUTED verification with reasoning includes notes in pending_compute_verdicts."""
        engine = self._make_engine()
        self._set_verification(
            engine,
            "WH-001",
            "REFUTED",
            summary="Expected 1/(8*pi*M) but got 1/(4*pi*M)",
        )
        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Verify formula X = Y",
        )
        engine._track_agent_result(task)

        assert len(engine._state.pending_compute_verdicts) == 1
        v = engine._state.pending_compute_verdicts[0]
        assert v["notes"] == "Expected 1/(8*pi*M) but got 1/(4*pi*M)"

    def test_context_suffix_renders_notes(self):
        """Context suffix renders notes when present in verdict dict."""
        engine = self._make_engine()
        engine._state.pending_compute_verdicts = [
            {
                "verdict": "REFUTED",
                "claim": "WH-001",
                "attempt": 1,
                "notes": "Factor of 2 discrepancy",
            }
        ]
        prefix = engine._build_context_suffix()
        assert "Notes: Factor of 2 discrepancy" in prefix

    def test_context_suffix_renders_failure_detail(self):
        """Context suffix renders notes when present."""
        engine = self._make_engine()
        engine._state.pending_compute_verdicts = [
            {
                "verdict": "REFUTED",
                "claim": "WH-001",
                "attempt": 1,
                "notes": "Division by zero at r=0",
            }
        ]
        prefix = engine._build_context_suffix()
        assert "Notes: Division by zero at r=0" in prefix

    def test_empty_notes_and_failure_detail_omitted(self):
        """Empty notes are omitted from suffix."""
        engine = self._make_engine()
        engine._state.pending_compute_verdicts = [
            {
                "verdict": "REFUTED",
                "claim": "WH-001",
                "attempt": 1,
                "notes": "",
            }
        ]
        prefix = engine._build_context_suffix()
        assert "Notes:" not in prefix

    def test_compute_verdict_signal_in_context_suffix(self):
        """Non-VERIFIED verdict appears in context suffix with attempt count."""
        engine = self._make_engine()
        self._set_verification(engine, "WH-001", "REFUTED")
        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Verify formula X = Y",
        )
        engine._track_agent_result(task)

        prefix = engine._build_context_suffix()
        assert "VERIFICATION RESULTS" in prefix
        assert "REFUTED" in prefix
        assert "Attempt 1/2" in prefix

    def test_empty_comp_log_noop(self):
        """No verification on target hypothesis, nothing happens."""
        engine = self._make_engine()
        from open_dirac.research_state import Hypothesis

        engine.research_state.hypotheses["WH-001"] = Hypothesis(id="WH-001")

        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Verify formula X = Y",
        )
        engine._track_agent_result(task)

        assert len(engine._state.claim_failure_count) == 0
        assert len(engine._state.pending_compute_verdicts) == 0


class TestRefutedEvidenceClearing:
    """Test that refuted evidence is cleared when new evidence is stored."""

    def test_store_evidence_clears_refuted(self):
        """New evidence replaces previously-refuted evidence on a hypothesis."""
        from open_dirac.agents.evidence_base import EvidenceAgent

        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            evidence=[
                Evidence(type="compute", result="wrong", iteration=1, refuted=True),
                Evidence(
                    type="compute", result="also wrong", iteration=2, refuted=True
                ),
            ],
        )

        # Create a minimal concrete subclass to test _store_evidence
        class _Stub(EvidenceAgent):
            def process_response(self, response, task, iteration):
                pass

            def build_context(self, task, iteration):
                return ""

        stub = _Stub.__new__(_Stub)
        stub.research_state = state

        new_ev = Evidence(type="compute", result="correct", iteration=3)
        stub._store_evidence("WH-001", new_ev)

        assert len(state.hypotheses["WH-001"].evidence) == 1
        assert state.hypotheses["WH-001"].evidence[0].result == "correct"
        assert not state.hypotheses["WH-001"].evidence[0].refuted

    def test_store_evidence_keeps_non_refuted(self):
        """Non-refuted evidence is preserved alongside new evidence."""
        from open_dirac.agents.evidence_base import EvidenceAgent

        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            evidence=[
                Evidence(type="compute", result="good", iteration=1, refuted=False),
                Evidence(type="compute", result="bad", iteration=2, refuted=True),
            ],
        )

        class _Stub(EvidenceAgent):
            def process_response(self, response, task, iteration):
                pass

            def build_context(self, task, iteration):
                return ""

        stub = _Stub.__new__(_Stub)
        stub.research_state = state

        new_ev = Evidence(type="compute", result="new", iteration=3)
        stub._store_evidence("WH-001", new_ev)

        assert len(state.hypotheses["WH-001"].evidence) == 2
        results = [ev.result for ev in state.hypotheses["WH-001"].evidence]
        assert "good" in results
        assert "new" in results
        assert "bad" not in results


class TestCriticCleanSignal:
    """Test that _no_critiques_filed flag injects a violation for the orchestrator."""

    def _make_engine(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 5
            engine._state = LoopState()
            engine.research_state = ResearchState()
            engine.critic = MagicMock()
        return engine

    def test_no_critiques_filed_produces_no_new_critiques(self):
        """_no_critiques_filed=True means no new critiques in research state."""
        engine = self._make_engine()
        response = MagicMock()
        response.text = ""
        engine.critic.run = MagicMock(return_value=response)
        engine.critic._no_critiques_filed = True

        task = Task(
            task_id="TASK-005", task_type=TaskType.CRITIQUE, assigned_to="deep_critic"
        )
        engine._dispatch(task)

        # No new critiques should have been filed
        recent = [
            c
            for c in engine.research_state.critiques.values()
            if c.iteration_filed == engine.iteration
        ]
        assert len(recent) == 0

    def test_normal_critique_no_violation(self):
        """_no_critiques_filed=False does NOT inject a violation."""
        engine = self._make_engine()
        response = MagicMock()
        response.text = "Critiques filed."
        engine.critic.run = MagicMock(return_value=response)
        engine.critic._no_critiques_filed = False

        task = Task(
            task_id="TASK-005", task_type=TaskType.CRITIQUE, assigned_to="deep_critic"
        )
        engine._dispatch(task)

        assert len(engine._state.pending_violations) == 0


class TestTerminationGate:
    """Test the termination gate in the main loop."""

    def _make_engine(self):
        """Create a OpenDirac instance for termination gate testing."""
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"
            written = {}

            def capture_write(filename, content):
                written[filename] = content

            ws.write_file = MagicMock(side_effect=capture_write)
            ws.read_file = MagicMock(return_value="")

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 5
            engine._state = LoopState()
            engine.research_state = ResearchState()
            engine.problem_meta = {}
        return engine, written

    def test_terminate_allowed_when_stub(self):
        """Stub can_terminate always returns True, so TERMINATE proceeds."""
        from open_dirac.validation import can_terminate

        engine, _ = self._make_engine()

        # The stub always allows termination
        allowed, blockers = can_terminate(
            engine.workspace,
            engine.config,
            engine.metrics,
            engine.problem_meta,
            research_state=engine.research_state,
        )
        assert allowed is True
        assert blockers == []

    def test_build_context_suffix_with_violations(self):
        """Context suffix includes pending violations."""
        engine, _ = self._make_engine()
        engine._state.pending_violations = [
            Violation(
                check="test_check",
                severity=ViolationSeverity.ERROR,
                message="Something wrong",
            ),
        ]
        prefix = engine._build_context_suffix()

        assert "POST-INTEGRATION VIOLATIONS" in prefix
        assert "test_check" in prefix
        assert "Something wrong" in prefix
        assert len(engine._state.pending_violations) == 0  # consumed

    def test_build_context_suffix_with_blockers(self):
        """Context suffix includes termination blockers."""
        engine, _ = self._make_engine()
        engine._state.pending_termination_blockers = [
            "Missing numerical verification",
            "Unresolved critiques remain",
        ]
        prefix = engine._build_context_suffix()

        assert "TERMINATION BLOCKED" in prefix
        assert "Missing numerical verification" in prefix
        assert "Unresolved critiques remain" in prefix
        assert len(engine._state.pending_termination_blockers) == 0  # consumed

    def test_termination_blockers_include_checklist(self):
        """Termination blockers banner includes the pre-dispatch checklist."""
        engine, _ = self._make_engine()
        engine._state.pending_termination_blockers = ["Some blocker"]
        prefix = engine._build_context_suffix()

        assert "Pre-dispatch checklist" in prefix
        assert "FILL IN placeholder" in prefix
        assert "closed-form SymPy" in prefix
        assert "MCQ answers" in prefix
        assert "Return types match" in prefix

    def test_build_context_suffix_empty_when_no_issues(self):
        """Context suffix is empty when no violations or blockers."""
        engine, _ = self._make_engine()
        prefix = engine._build_context_suffix()
        assert prefix == ""

    def test_context_suffix_includes_er_demotion_safety(self):
        """ER demotion safety violations now appear in context (no longer silently filtered)."""
        engine, _ = self._make_engine()
        engine._state.pending_violations = [
            Violation(
                check="er_demotion_safety",
                severity=ViolationSeverity.WARNING,
                message="ER-001 has REFUTED computation with no VERIFIED — demoted to WH-001",
                detail="ER-001",
            ),
            Violation(
                check="phantom_references",
                severity=ViolationSeverity.ERROR,
                message="Phantom reference COMP-999",
            ),
        ]
        prefix = engine._build_context_suffix()

        assert "POST-INTEGRATION VIOLATIONS" in prefix
        assert "phantom_references" in prefix
        assert "COMP-999" in prefix
        # ER demotion safety violations now appear in context suffix
        assert "er_demotion_safety" in prefix
        assert len(engine._state.pending_violations) == 0  # consumed


class TestCheckStatusField:
    """Test the renamed _check_status_field (formerly _should_terminate)."""

    def _make_engine(self, state_text=""):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"
            ws.read_file = MagicMock(return_value=state_text)

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.research_state = ResearchState()
            engine.workspace = ws
            engine.iteration = 0
        return engine

    def test_completed_status(self):
        engine = self._make_engine("status: completed\n# Problem")
        engine.research_state.status = "completed"
        assert engine._check_status_field() is True

    def test_abandoned_status(self):
        engine = self._make_engine('status: "abandoned"\n# Problem')
        engine.research_state.status = "abandoned"
        assert engine._check_status_field() is True

    def test_partially_complete_status(self):
        engine = self._make_engine("status: partially_complete\n# Problem")
        engine.research_state.status = "partially_complete"
        assert engine._check_status_field() is True

    def test_in_progress_status(self):
        engine = self._make_engine("status: in_progress\n# Problem")
        engine.research_state.status = "in_progress"
        assert engine._check_status_field() is False

    def test_empty_state(self):
        engine = self._make_engine("")
        assert engine._check_status_field() is False


class TestZeroOutputStallHandling:
    """Tests for zero-output stall detection and enrichment (Improvement 1C-1D)."""

    def _make_engine(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"
            written = {}

            def capture_write(filename, content):
                written[filename] = content

            ws.write_file = MagicMock(side_effect=capture_write)

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.metrics.last_critic_iteration = 4
            engine.iteration = 5
            engine._state = LoopState(last_content_iteration=5)
            engine.research_state = ResearchState()
        return engine, ws, written

    def test_enrich_flags_prior_failures(self):
        """Enrichment adds prior failure context when claim_failure_count > 0."""
        engine, ws, written = self._make_engine()
        engine._state.claim_failure_count["WH-003"] = 1
        ws.read_file = MagicMock(
            return_value="---\ntask_type: compute\n---\n\nCompute WH-003 mass."
        )
        task = Task(
            task_id="TASK-005",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            body="Compute WH-003 mass limit",
        )
        engine._enrich_compute_task_with_prior_failures(task)
        assert "CURRENT_TASK.md" in written
        enriched = written["CURRENT_TASK.md"]
        assert "Prior Failure Context" in enriched
        assert "1 prior failure(s)" in enriched


class TestDispatchNewAgents:
    """Test dispatch routing to new agents (researcher, computer, reviewer)."""

    def _make_engine(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 5
            engine._state = LoopState(last_content_iteration=5)
            engine.research_state = ResearchState()
            engine.researcher = MagicMock()
            engine.computer = MagicMock()
            engine.reviewer = MagicMock()
            engine.critic = MagicMock()
            engine.formatter = MagicMock(rejection_reason=None)
        return engine

    def test_research_dispatch(self):
        engine = self._make_engine()
        task = Task(
            task_id="TASK-005",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            iteration=5,
            body="Research WH-001",
        )
        name, _ = engine._dispatch(task)
        assert name == "researcher"

    def test_compute_dispatch(self):
        engine = self._make_engine()
        task = Task(
            task_id="TASK-005",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            iteration=5,
            body="Compute WH-001",
        )
        name, _ = engine._dispatch(task)
        assert name == "computer"

    def test_verify_dispatch(self):
        engine = self._make_engine()
        task = Task(
            task_id="TASK-005",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            iteration=5,
            body="Verify WH-001",
        )
        name, _ = engine._dispatch(task)
        assert name == "reviewer"

    def test_verify_routes_correctly(self):
        engine = self._make_engine()
        task = Task(
            task_id="TASK-005",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            iteration=5,
            body="Verify something",
        )
        name, _ = engine._dispatch(task)
        assert name == "reviewer"


class TestUpdateResearchIteration:
    """Test engine-side iteration counter update."""

    def test_iteration_field_updated(self):
        from open_dirac.engine import OpenDirac

        engine = OpenDirac.__new__(OpenDirac)
        engine.config = Config()
        engine.research_state = ResearchState()
        engine.iteration = 5
        engine._update_research_iteration()
        assert engine.research_state.iteration == 5

    def test_iteration_starts_at_zero(self):
        from open_dirac.engine import OpenDirac

        engine = OpenDirac.__new__(OpenDirac)
        engine.config = Config()
        engine.research_state = ResearchState()
        assert engine.research_state.iteration == 0
        engine.iteration = 1
        engine._update_research_iteration()
        assert engine.research_state.iteration == 1


class TestDispatchFailureRecovery:
    """Test that transient dispatch failures are caught and the loop continues."""

    def _make_engine(self):
        """Create a OpenDirac instance with mocked agents for dispatch testing."""
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"
            ws.read_file = MagicMock(return_value="")
            ws.write_file = MagicMock()
            ws.file_size = MagicMock(return_value=0)
            ws.git_commit = MagicMock()

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config(max_iterations=3)
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.metrics.last_critic_iteration = 0
            engine.metrics.alerts = []
            engine.metrics.calls = []
            engine.metrics.total_input_tokens = 0
            engine.metrics.total_output_tokens = 0
            engine.iteration = 0
            engine._state = LoopState()
            engine.research_state = ResearchState()
            engine.problem_meta = {}

            engine.orchestrator = MagicMock()
            engine.researcher = MagicMock()
            engine.computer = MagicMock()
            engine.reviewer = MagicMock()
            engine.critic = MagicMock()
            engine.formatter = MagicMock(rejection_reason=None)
            engine.surveyor = MagicMock()
            engine.surveyor.parsed_survey = None
            engine.planner = MagicMock()
            engine.planner.parsed_strategy = None
        return engine

    def test_transient_error_continues_loop(self):
        """A transient 504 from dispatch is caught; loop continues to next iteration."""
        engine = self._make_engine()
        engine.config.max_iterations = 10  # enough headroom to avoid budget override

        # Create a 504-like exception
        exc_504 = Exception("Server error")
        exc_504.status_code = 504

        # Orchestrator returns a verify task each time
        task = Task(
            task_id="TASK-001",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            iteration=1,
            body="Verify something.",
        )
        engine.orchestrator.parse_task = MagicMock(return_value=task)

        # First dispatch: transient error; second+third: succeed, then terminate
        task_terminate = Task(
            task_id="TASK-002",
            task_type=TaskType.TERMINATE,
            assigned_to="orchestrator",
            iteration=2,
        )
        engine.orchestrator.parse_task = MagicMock(side_effect=[task, task_terminate])
        engine.reviewer.run = MagicMock(side_effect=exc_504)

        engine.run()

        # Alert was logged for the failure
        engine.metrics.alert.assert_any_call(
            1, unittest_any_string_containing("Dispatch failed")
        )
        # reviewer was called once (failed), then orchestrator terminated
        assert engine.reviewer.run.call_count == 1
        assert engine.iteration == 2

    def test_non_recoverable_error_propagates(self):
        """A non-recoverable error (e.g. RuntimeError) propagates and crashes.

        Note: TypeError/ValueError/KeyError/AttributeError are treated as
        recoverable (likely malformed LLM output) and skip the iteration.
        """
        engine = self._make_engine()

        task = Task(
            task_id="TASK-001",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            iteration=1,
            body="Research something.",
        )
        engine.orchestrator.parse_task = MagicMock(return_value=task)
        engine.researcher.run = MagicMock(side_effect=RuntimeError("fatal bug"))

        import pytest

        with pytest.raises(RuntimeError, match="fatal bug"):
            engine.run()

    def test_llm_data_error_skips_iteration(self):
        """TypeError/ValueError from malformed LLM output skip the iteration."""
        engine = self._make_engine()

        task = Task(
            task_id="TASK-001",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            iteration=1,
            body="Research something.",
        )
        engine.orchestrator.parse_task = MagicMock(return_value=task)
        engine.researcher.run = MagicMock(
            side_effect=TypeError("expected string, got list")
        )

        # Should NOT raise — the engine skips the iteration
        engine.run()

    def test_dispatch_failure_injects_violation(self):
        """Transient dispatch failure adds a violation for the orchestrator."""
        engine = self._make_engine()
        engine.config.max_iterations = 10

        exc_timeout = Exception("Read timed out")
        exc_timeout.status_code = 504

        task = Task(
            task_id="TASK-001",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            iteration=1,
            body="Verify something.",
        )
        task_terminate = Task(
            task_id="TASK-002",
            task_type=TaskType.TERMINATE,
            assigned_to="orchestrator",
            iteration=2,
        )
        engine.orchestrator.parse_task = MagicMock(side_effect=[task, task_terminate])
        engine.reviewer.run = MagicMock(side_effect=exc_timeout)

        engine.run()

        # Violation was consumed by _build_context_suffix on iteration 2, but
        # we can verify it was created by checking the context_suffix set on orchestrator
        prefix = engine.orchestrator.context_suffix
        assert "dispatch_failure" in prefix


class TestAgentFailureRouting:
    """Test _record_agent_failures and its integration with _build_context_suffix."""

    def _make_engine(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 5
            engine._state = LoopState()
            engine.research_state = ResearchState()
        return engine

    def test_max_tokens_recorded(self):
        """max_tokens stop_reason records a truncation failure with token count."""
        engine = self._make_engine()
        result = MagicMock()
        result.stop_reason = "max_tokens"
        result.output_tokens = 8000
        task = Task(
            task_id="TASK-005", task_type=TaskType.RESEARCH, assigned_to="researcher"
        )

        engine._record_agent_failures(task, "researcher", result)

        assert len(engine._state.agent_failures) == 1
        assert engine._state.agent_failures[0]["event"] == "max_tokens_truncation"
        assert engine._state.agent_failures[0]["task_id"] == "TASK-005"
        assert engine._state.agent_failures[0]["agent"] == "researcher"
        assert "8000 tokens" in engine._state.agent_failures[0]["detail"]
        assert "Decompose" in engine._state.agent_failures[0]["detail"]

    def test_max_rounds_forced_recorded(self):
        """max_rounds_forced stop_reason records an exhaustion failure."""
        from open_dirac.llm import AgentResult

        engine = self._make_engine()
        result = AgentResult(text="partial", rounds=10, stop_reason="max_rounds_forced")
        task = Task(
            task_id="TASK-005", task_type=TaskType.REVIEW, assigned_to="reviewer"
        )

        engine._record_agent_failures(task, "reviewer", result)

        assert len(engine._state.agent_failures) == 1
        assert engine._state.agent_failures[0]["event"] == "max_rounds_exhaustion"
        assert "10 tool-use rounds" in engine._state.agent_failures[0]["detail"]

    def test_normal_end_turn_not_recorded(self):
        """Normal end_turn does NOT record a failure."""
        engine = self._make_engine()
        result = MagicMock()
        result.stop_reason = "end_turn"
        task = Task(
            task_id="TASK-005", task_type=TaskType.RESEARCH, assigned_to="researcher"
        )

        engine._record_agent_failures(task, "researcher", result)

        assert len(engine._state.agent_failures) == 0

    def test_context_suffix_includes_agent_failures(self):
        """Agent failures appear in context suffix banner."""
        engine = self._make_engine()
        engine._state.agent_failures = [
            {
                "task_id": "TASK-004",
                "agent": "researcher",
                "event": "max_tokens_truncation",
                "detail": (
                    "Output hit token limit (8000 tokens). "
                    "Decompose into smaller subtasks, each targeting a single "
                    "derivation step or sub-claim."
                ),
                "iteration": 4,
            }
        ]
        prefix = engine._build_context_suffix()

        assert "AGENT FAILURES" in prefix
        assert "TASK-004" in prefix
        assert "max_tokens_truncation" in prefix
        assert "Decompose" in prefix
        assert "token limit" in prefix

    def test_context_suffix_clears_agent_failures(self):
        """Agent failures are cleared after building context suffix."""
        engine = self._make_engine()
        engine._state.agent_failures = [
            {
                "task_id": "TASK-004",
                "agent": "researcher",
                "event": "max_tokens_truncation",
                "detail": "Task too large.",
                "iteration": 4,
            }
        ]
        engine._build_context_suffix()
        assert len(engine._state.agent_failures) == 0

    def test_compute_verdict_appends_to_pending_verdicts(self):
        """REFUTED verdict below stall limit appends to pending_compute_verdicts."""
        from open_dirac.research_state import Hypothesis, ReviewResult

        engine = self._make_engine()
        engine.iteration = 5
        engine.research_state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        engine.research_state.hypotheses["WH-001"].review = ReviewResult(
            verdict="REFUTED",
            summary="Mismatch",
            iteration=5,
        )

        task = Task(
            task_id="TASK-005",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Verify formula X = Y",
        )
        engine._track_agent_result(task)

        # Verdict now goes to pending_compute_verdicts, not agent_failures
        assert len(engine._state.pending_compute_verdicts) == 1
        assert engine._state.pending_compute_verdicts[0]["verdict"] == "REFUTED"
        assert engine._state.pending_compute_verdicts[0]["attempt"] == 1

    def test_compute_verdict_stall_signals_orchestrator(self):
        """At stall (count >= limit), verdict signal still goes to pending_compute_verdicts."""
        from open_dirac.research_state import Hypothesis, ReviewResult

        engine = self._make_engine()
        engine.iteration = 5
        engine.config.stall_recompute_limit = 2
        engine._state.claim_failure_count["WH-001"] = 1  # already at limit-1
        engine.research_state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        engine.research_state.hypotheses["WH-001"].review = ReviewResult(
            verdict="INCONCLUSIVE",
            summary="Unclear",
            iteration=5,
        )

        task = Task(
            task_id="TASK-005",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Verify formula X = Y",
        )
        engine._track_agent_result(task)

        assert len(engine._state.pending_compute_verdicts) == 1
        assert engine._state.pending_compute_verdicts[0]["attempt"] == 2

    def test_context_suffix_ordering(self):
        """Violations appear before agent failures in context suffix."""
        engine = self._make_engine()
        engine._state.pending_violations = [
            Violation(
                check="test",
                severity=ViolationSeverity.WARNING,
                message="test violation",
            ),
        ]
        engine._state.agent_failures = [
            {
                "task_id": "TASK-003",
                "agent": "reviewer",
                "event": "max_rounds_exhaustion",
                "detail": "Exhausted 10 tool-use rounds without completing.",
                "iteration": 4,
            }
        ]
        prefix = engine._build_context_suffix()

        violations_pos = prefix.index("VIOLATIONS")
        failures_pos = prefix.index("AGENT FAILURES")
        assert violations_pos < failures_pos


class TestProblemStatementPopulated:
    """Test that ResearchState gets problem_statement populated on init (A1)."""

    def test_problem_statement_set_on_init(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.name = "20260316_test_run"
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 0
            engine._state = LoopState()
            # Simulate what __init__ does after workspace.init
            problem = "Derive the Hawking temperature."
            engine.research_state = ResearchState()
            engine.research_state.problem_statement = problem.strip()
            engine.research_state.title = ws.root.name

            assert (
                engine.research_state.problem_statement
                == "Derive the Hawking temperature."
            )
            assert engine.research_state.title == "20260316_test_run"

    def test_answer_template_stored_separately(self):
        """answer_template is stored as a separate field, not embedded in problem_statement."""
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.name = "test_run"
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"

            from open_dirac.engine import OpenDirac

            engine = OpenDirac(
                "Derive T_H.", Config(), answer_template="## Answer\n\nT_H = ?"
            )

            assert engine.research_state.problem_statement == "Derive T_H."
            assert engine.research_state.answer_template == "## Answer\n\nT_H = ?"


class TestSyncOnTermination:
    """Test that _sync_research_state is called on termination path (A3)."""

    def _make_engine(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"
            ws.read_file = MagicMock(return_value="")
            ws.write_file = MagicMock()
            ws.file_size = MagicMock(return_value=0)
            ws.git_commit = MagicMock()

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config(max_iterations=3)
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.metrics.last_critic_iteration = 0
            engine.metrics.alerts = []
            engine.metrics.calls = []
            engine.metrics.total_input_tokens = 0
            engine.metrics.total_output_tokens = 0
            engine.iteration = 0
            engine._state = LoopState()
            engine.research_state = ResearchState()
            engine.problem_meta = {}
            engine.orchestrator = MagicMock()
            engine.researcher = MagicMock()
            engine.computer = MagicMock()
            engine.reviewer = MagicMock()
            engine.critic = MagicMock()
            engine.formatter = MagicMock(rejection_reason=None)
            engine.surveyor = MagicMock()
            engine.surveyor.parsed_survey = None
            engine.planner = MagicMock()
            engine.planner.parsed_strategy = None
        return engine, ws

    def test_sync_called_on_termination(self):
        """_sync_research_state is called when termination is allowed."""
        engine, ws = self._make_engine()
        engine.config.max_iterations = 10

        task_terminate = Task(
            task_id="TASK-001",
            task_type=TaskType.TERMINATE,
            assigned_to="orchestrator",
            iteration=1,
        )
        engine.orchestrator.parse_task = MagicMock(return_value=task_terminate)

        with patch.object(engine, "_sync_research_state") as mock_sync:
            engine.run()
            mock_sync.assert_called()


class TestExploreResultSuppression:
    """Test that evidence-less results are NOT appended to pending_explore_results (C3)."""

    def _make_engine(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 3
            engine._state = LoopState()
            engine.research_state = ResearchState()
        return engine

    def test_no_evidence_not_appended(self):
        """Hypothesis with no evidence NOT appended to pending_explore_results."""
        from open_dirac.research_state import Hypothesis

        engine = self._make_engine()
        engine.research_state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        task = Task(
            task_id="TASK-003",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            target_claim="WH-001",
            body="Research WH-001",
        )
        engine._track_agent_result(task)

        assert len(engine._state.pending_explore_results) == 0

    def test_empty_result_explore_not_appended(self):
        """Evidence with empty result NOT appended to pending_explore_results."""
        from open_dirac.research_state import Hypothesis, Evidence

        engine = self._make_engine()
        engine.research_state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            evidence=[Evidence(result="", method="test")],
        )
        task = Task(
            task_id="TASK-003",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            target_claim="WH-001",
            body="Compute WH-001",
        )
        engine._track_agent_result(task)

        assert len(engine._state.pending_explore_results) == 0

    def test_successful_explore_appended(self):
        """Successful evidence IS appended to pending_explore_results."""
        from open_dirac.research_state import Hypothesis, Evidence

        engine = self._make_engine()
        engine.research_state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            evidence=[
                Evidence(result="x = 42", method="Compute x", confidence="exact")
            ],
        )
        task = Task(
            task_id="TASK-003",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            target_claim="WH-001",
            body="Compute WH-001",
        )
        engine._track_agent_result(task)

        assert len(engine._state.pending_explore_results) == 1
        assert engine._state.pending_explore_results[0]["target_id"] == "WH-001"

    def test_critique_evidence_appended(self):
        """Evidence on a critique target IS appended to pending_explore_results."""
        from open_dirac.research_state import (
            Critique,
            Evidence,
            Severity,
            CritiqueStatus,
        )

        engine = self._make_engine()
        engine.research_state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.HIGH,
            argument="Spin prediction may be wrong.",
            status=CritiqueStatus.ACTIVE,
            iteration_filed=2,
            evidence=[
                Evidence(
                    type="research",
                    method="re-derivation",
                    result="Spin is indeed 1",
                    confidence="exact",
                )
            ],
        )
        task = Task(
            task_id="TASK-006",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            target_claim="CRIT-001",
            body="Investigate CRIT-001",
        )
        engine._track_agent_result(task)

        assert len(engine._state.pending_explore_results) == 1
        r = engine._state.pending_explore_results[0]
        assert r["target_id"] == "CRIT-001"
        assert r["task_type"] == "research"

    def test_critique_no_evidence_not_appended(self):
        """Critique with no evidence NOT appended to pending_explore_results."""
        from open_dirac.research_state import Critique, Severity, CritiqueStatus

        engine = self._make_engine()
        engine.research_state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.HIGH,
            argument="Spin prediction may be wrong.",
            status=CritiqueStatus.ACTIVE,
            iteration_filed=2,
        )
        task = Task(
            task_id="TASK-006",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            target_claim="CRIT-001",
            body="Investigate CRIT-001",
        )
        engine._track_agent_result(task)

        assert len(engine._state.pending_explore_results) == 0


def unittest_any_string_containing(substring):
    """Helper matcher: matches any string containing the given substring."""

    class _Matcher:
        def __eq__(self, other):
            return isinstance(other, str) and substring in other

        def __repr__(self):
            return f"<string containing {substring!r}>"

    return _Matcher()


class TestCallWithRetryNoRetry:
    """_call_with_retry no longer retries on max_tokens — continuation is
    handled upstream by continue_on_max_tokens in call_llm. This class tests
    that a still-truncated response (after upstream continuation) is surfaced
    correctly (alert + scaffold event) without triggering a second retry here.
    """

    def _make_agent(self, tmp_path):
        """Create a minimal concrete agent for testing _call_with_retry."""
        from open_dirac.agents.base import BaseAgent

        class _StubAgent(BaseAgent):
            name = "test_agent"
            prompt_file = ""

            def __init__(self, config, workspace, metrics):
                super().__init__(config, workspace, metrics)
                self._system_prompt = "system"

            def build_context(self, task, iteration):
                return "context"

            def process_response(self, response, task, iteration):
                pass

        config = Config(workspace_dir=str(tmp_path))
        ws = MagicMock()
        ws.root = tmp_path
        from open_dirac.metrics import MetricsTracker

        metrics = MetricsTracker()
        return _StubAgent(config, ws, metrics), metrics

    def test_no_retry_on_max_tokens(self, tmp_path):
        """call_llm is invoked exactly once even when it returns max_tokens."""
        from open_dirac.llm import LLMResponse

        agent, metrics = self._make_agent(tmp_path)

        response = LLMResponse(
            text="partial output...",
            input_tokens=5000,
            output_tokens=8000,
            stop_reason="max_tokens",
            duration=1.0,
        )
        with patch(
            "open_dirac.agents.base.call_llm", return_value=response
        ) as mock_llm:
            result = agent._call_with_retry("long context", iteration=3)

        mock_llm.assert_called_once()
        assert result.stop_reason == "max_tokens"
        assert result.output_tokens == 8000

    def test_normal_stop_returns_immediately(self, tmp_path):
        """Normal end_turn returns without alert or scaffold event."""
        from open_dirac.llm import LLMResponse

        agent, metrics = self._make_agent(tmp_path)

        response = LLMResponse(
            text="full output",
            input_tokens=3000,
            output_tokens=2000,
            stop_reason="end_turn",
            duration=0.5,
        )
        with patch(
            "open_dirac.agents.base.call_llm", return_value=response
        ) as mock_llm:
            result = agent._call_with_retry("context", iteration=1)

        mock_llm.assert_called_once()
        assert result.stop_reason == "end_turn"
        assert len(metrics.alerts) == 0

    def test_max_tokens_fires_alert_and_scaffold_event(self, tmp_path):
        """max_tokens triggers a metrics alert and scaffold log event."""
        from open_dirac.llm import LLMResponse

        agent, metrics = self._make_agent(tmp_path)

        response = LLMResponse(
            text="truncated...",
            input_tokens=5000,
            output_tokens=8000,
            stop_reason="max_tokens",
            duration=1.0,
        )
        with (
            patch("open_dirac.agents.base.call_llm", return_value=response),
            patch("open_dirac.agents.base.log_scaffold_event") as mock_log,
        ):
            agent._call_with_retry("context", iteration=3)

        # Alert fired
        assert len(metrics.alerts) == 1
        assert "max_tokens_reached" in metrics.alerts[0]["message"]

        # Scaffold event emitted
        mock_log.assert_called_once()
        args, kwargs = mock_log.call_args
        assert args[1] == 3  # iteration
        assert kwargs["category"] == "loop_control"
        assert kwargs["event"] == "max_tokens_unrecoverable"
        assert "test_agent" in kwargs["detail"]


# ===========================================================================
# Surveyor integration
# ===========================================================================


class TestSurveyorEngine:
    """Tests for surveyor agent integration in the engine."""

    def _make_engine(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"
            ws.write_file = MagicMock()
            ws.git_commit = MagicMock()
            ws.file_size = MagicMock(return_value=0)

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.research_state = ResearchState(
                problem_statement="Derive Hawking temperature.",
            )
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 0
            engine._state = LoopState()

            # Create mock surveyor
            engine.surveyor = MagicMock()
            engine.surveyor.parsed_survey = None
            engine.planner = MagicMock()
            engine.planner.parsed_strategy = None
        return engine

    def test_apply_survey_stores_survey(self):
        engine = self._make_engine()

        engine.surveyor.parsed_survey = {
            "raw_notes": "Derive kappa via Killing vectors first.",
        }
        engine._apply_survey()

        assert engine.research_state.survey_background != ""
        assert "Killing vectors" in engine.research_state.survey_background

    def test_apply_survey_none_does_nothing(self):
        engine = self._make_engine()
        engine.surveyor.parsed_survey = None
        engine._apply_survey()
        assert engine.research_state.survey_background == ""


class TestTerminationCircuitBreaker:
    """Test the circuit breaker that auto-abandons WHs after repeated termination blocks."""

    def _make_engine(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"
            ws.write_file = MagicMock()

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.research_state = ResearchState()
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 10
            engine._state = LoopState()
        return engine

    def test_force_abandon_working_hypotheses(self):
        """Auto-abandon all remaining WHs when circuit breaker fires."""
        from open_dirac.research_state import Hypothesis, HypothesisStatus

        engine = self._make_engine()
        engine.research_state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
            statement="Claim A",
        )
        engine.research_state.hypotheses["WH-004"] = Hypothesis(
            id="WH-004",
            status=HypothesisStatus.WORKING,
            statement="Claim B",
        )
        engine.research_state.hypotheses["ER-002"] = Hypothesis(
            id="ER-002",
            status=HypothesisStatus.ESTABLISHED,
        )
        engine._state.consecutive_termination_blocks = 3

        engine._force_abandon_working_hypotheses()

        assert (
            engine.research_state.hypotheses["WH-001"].status
            == HypothesisStatus.ABANDONED
        )
        assert (
            engine.research_state.hypotheses["WH-004"].status
            == HypothesisStatus.ABANDONED
        )
        assert (
            engine.research_state.hypotheses["ER-002"].status
            == HypothesisStatus.ESTABLISHED
        )
        assert len(engine.research_state.failed_approaches) == 2
        assert engine._state.consecutive_termination_blocks == 0

    def test_counter_increments_on_block(self):
        """consecutive_termination_blocks increments when termination is blocked."""
        engine = self._make_engine()
        assert engine._state.consecutive_termination_blocks == 0
        engine._state.consecutive_termination_blocks += 1
        assert engine._state.consecutive_termination_blocks == 1

    def test_counter_resets_on_non_terminate(self):
        """The counter should reset when a non-terminate task is processed."""
        engine = self._make_engine()
        engine._state.consecutive_termination_blocks = 2
        # Simulate what engine does for non-terminate tasks
        engine._state.consecutive_termination_blocks = 0
        assert engine._state.consecutive_termination_blocks == 0

    def test_no_abandon_when_no_working_hypotheses(self):
        """Force-abandon is a no-op when all hypotheses are already established/abandoned."""
        from open_dirac.research_state import Hypothesis, HypothesisStatus

        engine = self._make_engine()
        engine.research_state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        engine._state.consecutive_termination_blocks = 5

        engine._force_abandon_working_hypotheses()

        assert (
            engine.research_state.hypotheses["ER-001"].status
            == HypothesisStatus.ESTABLISHED
        )
        assert len(engine.research_state.failed_approaches) == 0


class TestRedundantCriticPassFix:
    """Test that forced critic clears stale blockers and injects can-terminate signal."""

    def _make_engine(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 5
            engine._state = LoopState()
            engine.research_state = ResearchState()
            engine.critic = MagicMock()
        return engine

    def test_should_trigger_critic_after_verified_review(self):
        """_should_trigger_critic returns True after any VERIFIED review."""
        engine = self._make_engine()
        engine.metrics.last_critic_iteration = 0
        engine._state.last_verified_review_iteration = 5

        assert engine._should_trigger_critic() is True

    def test_should_trigger_critic_even_when_recent(self):
        """Critic fires on every ER — no delay constraint."""
        engine = self._make_engine()
        engine.metrics.last_critic_iteration = 4  # ran just 1 iteration ago
        engine._state.last_verified_review_iteration = 5

        assert engine._should_trigger_critic() is True

    def test_should_not_trigger_critic_without_verified_review(self):
        """_should_trigger_critic returns False when no verified review this iteration."""
        engine = self._make_engine()
        engine.metrics.last_critic_iteration = 0
        engine._state.last_verified_review_iteration = (
            3  # different from current iteration
        )

        assert engine._should_trigger_critic() is False


class TestDispatchHistory:
    """Test dispatch history recording and rendering in orchestrator context."""

    def _make_engine(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 3
            engine._state = LoopState()
            engine.research_state = ResearchState()
            engine.orchestrator = MagicMock()
            engine.orchestrator.dispatch_history_text = ""
        return engine

    def test_dispatch_history_renders_on_orchestrator(self):
        """DispatchRecords render in orchestrator.dispatch_history_text."""
        engine = self._make_engine()
        engine._state.dispatch_history = [
            DispatchRecord(
                iteration=1,
                task_type="compute",
                target="RQ-001",
                outcome="evidence (exact)",
            ),
            DispatchRecord(
                iteration=2, task_type="review", target="WH-001", outcome="REFUTED"
            ),
        ]
        engine._build_context_suffix()
        dh = engine.orchestrator.dispatch_history_text
        assert "<tasks_dispatch_history>" in dh
        assert "Iter 1: compute → RQ-001 | evidence (exact)" in dh
        assert "Iter 2: review → WH-001 | REFUTED" in dh
        assert "</tasks_dispatch_history>" in dh

    def test_dispatch_history_persists_across_calls(self):
        """Dispatch history is NOT consumed — persists across _build_context_suffix calls."""
        engine = self._make_engine()
        engine._state.dispatch_history = [
            DispatchRecord(
                iteration=1,
                task_type="compute",
                target="RQ-001",
                outcome="evidence (exact)",
            ),
        ]
        engine._build_context_suffix()
        dh1 = engine.orchestrator.dispatch_history_text
        engine._build_context_suffix()
        dh2 = engine.orchestrator.dispatch_history_text
        assert "tasks_dispatch_history" in dh1
        assert "tasks_dispatch_history" in dh2
        assert len(engine._state.dispatch_history) == 1

    def test_dispatch_history_empty_no_section(self):
        """No dispatch history → no dispatch_history_text set."""
        engine = self._make_engine()
        engine._build_context_suffix()
        assert engine.orchestrator.dispatch_history_text == ""

    def test_dispatch_history_no_target_omits_arrow(self):
        """Records with no target omit the arrow."""
        engine = self._make_engine()
        engine._state.dispatch_history = [
            DispatchRecord(
                iteration=4, task_type="critique", target=None, outcome="3 critique(s)"
            ),
        ]
        engine._build_context_suffix()
        dh = engine.orchestrator.dispatch_history_text
        assert "Iter 4: critique | 3 critique(s)" in dh
        assert "→" not in dh.split("Iter 4:")[1].split("|")[0]

    def test_append_dispatch_record_compute_with_evidence(self):
        """Compute task with evidence on target RQ records confidence."""
        engine = self._make_engine()
        from open_dirac.research_state import ResearchQuestion, Evidence

        engine.research_state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="What is T?",
        )
        engine.research_state.research_questions["RQ-001"].evidence = [
            Evidence(
                type="compute",
                result="T = 1/(8*pi*M)",
                confidence="exact",
                iteration=1,
            )
        ]
        task = Task(
            task_id="TASK-001",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            target_claim="RQ-001",
        )
        engine._append_dispatch_record(task)

        assert len(engine._state.dispatch_history) == 1
        rec = engine._state.dispatch_history[0]
        assert rec.task_type == "compute"
        assert rec.target == "RQ-001"
        assert rec.outcome == "evidence (exact)"

    def test_append_dispatch_record_compute_no_evidence(self):
        """Compute task with no evidence on target records 'no evidence'."""
        engine = self._make_engine()
        from open_dirac.research_state import ResearchQuestion

        engine.research_state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="What is T?",
        )
        task = Task(
            task_id="TASK-001",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            target_claim="RQ-001",
        )
        engine._append_dispatch_record(task)

        assert engine._state.dispatch_history[0].outcome == "no evidence"

    def test_append_dispatch_record_research_on_hypothesis(self):
        """Research task targeting a WH reads evidence from hypothesis."""
        engine = self._make_engine()
        from open_dirac.research_state import Hypothesis, Evidence

        engine.research_state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            evidence=[
                Evidence(
                    type="research",
                    result="derived",
                    confidence="approximate",
                    iteration=2,
                )
            ],
        )
        task = Task(
            task_id="TASK-002",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            target_claim="WH-001",
        )
        engine._append_dispatch_record(task)

        assert engine._state.dispatch_history[0].outcome == "evidence (approximate)"

    def test_append_dispatch_record_review_verdict(self):
        """Review task captures the reviewer's verdict."""
        engine = self._make_engine()
        from open_dirac.research_state import Hypothesis, ReviewResult

        engine.research_state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            review=ReviewResult(verdict="VERIFIED", summary="Correct.", iteration=3),
        )
        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
        )
        engine._append_dispatch_record(task)

        rec = engine._state.dispatch_history[0]
        assert rec.task_type == "review"
        assert rec.target == "WH-001"
        assert rec.outcome == "VERIFIED → WH-001"

    def test_append_dispatch_record_review_promoted_wh(self):
        """Review task finds promoted WH via ER- fallback."""
        engine = self._make_engine()
        from open_dirac.research_state import Hypothesis, ReviewResult, HypothesisStatus

        # Simulate post-promotion state: WH-001 gone, ER-001 present
        engine.research_state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
            review=ReviewResult(verdict="VERIFIED", summary="OK.", iteration=2),
        )
        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
        )
        engine._append_dispatch_record(task)

        rec = engine._state.dispatch_history[0]
        assert rec.outcome == "VERIFIED → ER-001"

    def test_append_dispatch_record_review_no_review(self):
        """Review task with no review result records 'no review produced'."""
        engine = self._make_engine()
        from open_dirac.research_state import Hypothesis

        engine.research_state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        task = Task(
            task_id="TASK-003",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
        )
        engine._append_dispatch_record(task)

        assert engine._state.dispatch_history[0].outcome == "no review produced"

    def test_append_dispatch_record_critique_with_critiques(self):
        """Critique task counts recent critiques from research state."""
        from open_dirac.research_state import Critique, Severity, CritiqueStatus

        engine = self._make_engine()
        # Add 3 critiques filed this iteration
        for i in range(1, 4):
            engine.research_state.critiques[f"CRIT-{i:03d}"] = Critique(
                id=f"CRIT-{i:03d}",
                severity=Severity.HIGH,
                status=CritiqueStatus.ACTIVE,
                targets=["WH-001"],
                argument=f"Issue {i}",
                iteration_filed=engine.iteration,
            )
        task = Task(
            task_id="TASK-004", task_type=TaskType.CRITIQUE, assigned_to="deep_critic"
        )
        engine._append_dispatch_record(task)

        rec = engine._state.dispatch_history[0]
        assert rec.task_type == "critique"
        assert rec.target is None
        assert rec.outcome == "3 critique(s)"

    def test_append_dispatch_record_critique_clean(self):
        """Critique task with no critiques filed records 'no critiques'."""
        engine = self._make_engine()
        # No critiques in research state
        task = Task(
            task_id="TASK-004", task_type=TaskType.CRITIQUE, assigned_to="deep_critic"
        )
        engine._append_dispatch_record(task)

        assert engine._state.dispatch_history[0].outcome == "no critiques"

    def test_append_dispatch_record_terminate_blocked(self):
        """Terminate task records 'blocked'."""
        engine = self._make_engine()
        task = Task(
            task_id="TASK-005", task_type=TaskType.TERMINATE, assigned_to="orchestrator"
        )
        engine._append_dispatch_record(task)

        rec = engine._state.dispatch_history[0]
        assert rec.task_type == "terminate"
        assert rec.outcome == "blocked"

    def test_append_dispatch_record_format_completed(self):
        """Format task records 'completed'."""
        engine = self._make_engine()
        task = Task(
            task_id="FORMAT-003", task_type=TaskType.FORMAT, assigned_to="formatter"
        )
        engine._append_dispatch_record(task)

        assert engine._state.dispatch_history[0].outcome == "completed"

    def test_dispatch_history_separate_from_violations(self):
        """Dispatch history is set on orchestrator, not in suffix with violations."""
        engine = self._make_engine()
        engine._state.dispatch_history = [
            DispatchRecord(
                iteration=1,
                task_type="compute",
                target="RQ-001",
                outcome="evidence (exact)",
            ),
        ]
        engine._state.pending_violations = [
            Violation(check="test", severity=ViolationSeverity.WARNING, message="oops"),
        ]
        suffix = engine._build_context_suffix()
        # Dispatch history is NOT in the suffix anymore
        assert "tasks_dispatch_history" not in suffix
        # But IS set on the orchestrator
        assert "tasks_dispatch_history" in engine.orchestrator.dispatch_history_text
        # Violations are still in the suffix
        assert "POST-INTEGRATION VIOLATIONS" in suffix

    def test_dispatch_history_capped_to_recent_iterations(self):
        """Old dispatch records are omitted, with a summary line."""
        engine = self._make_engine()
        engine.iteration = 10
        engine._state.dispatch_history = [
            DispatchRecord(
                iteration=1,
                task_type="compute",
                target="RQ-001",
                outcome="evidence (exact)",
            ),
            DispatchRecord(
                iteration=3, task_type="review", target="WH-001", outcome="VERIFIED"
            ),
            DispatchRecord(
                iteration=6,
                task_type="compute",
                target="RQ-002",
                outcome="evidence (approx)",
            ),
            DispatchRecord(
                iteration=8, task_type="review", target="WH-002", outcome="REFUTED"
            ),
            DispatchRecord(
                iteration=10,
                task_type="compute",
                target="RQ-003",
                outcome="evidence (exact)",
            ),
        ]
        engine._build_context_suffix()
        dh = engine.orchestrator.dispatch_history_text
        # Recent records (iter >= 6) are present
        assert "Iter 6" in dh
        assert "Iter 8" in dh
        assert "Iter 10" in dh
        # Old records (iter < 6) are omitted
        assert "Iter 1:" not in dh
        assert "Iter 3:" not in dh
        # Summary line present
        assert "2 earlier dispatch(es) omitted" in dh

    def test_dispatch_history_no_omission_early(self):
        """At early iterations, all records are included without omission line."""
        engine = self._make_engine()
        engine.iteration = 3
        engine._state.dispatch_history = [
            DispatchRecord(
                iteration=1, task_type="compute", target="RQ-001", outcome="evidence"
            ),
            DispatchRecord(
                iteration=2, task_type="review", target="WH-001", outcome="VERIFIED"
            ),
            DispatchRecord(
                iteration=3, task_type="compute", target="RQ-002", outcome="evidence"
            ),
        ]
        engine._build_context_suffix()
        dh = engine.orchestrator.dispatch_history_text
        assert "Iter 1" in dh
        assert "Iter 2" in dh
        assert "Iter 3" in dh
        assert "omitted" not in dh


class TestAutoPromoteCascade:
    """Tests for cascading auto-promotion in _auto_promote."""

    def _make_engine(self):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 5
            engine._state = LoopState()
            engine.research_state = ResearchState()
        return engine

    def test_simple_promotion(self):
        """VERIFIED WH with no deps is promoted."""
        from open_dirac.research_state import Hypothesis, HypothesisStatus, Verdict

        engine = self._make_engine()
        engine.research_state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict=Verdict.VERIFIED, summary="OK", iteration=4),
        )
        engine._auto_promote("WH-001")
        assert "ER-001" in engine.research_state.hypotheses
        assert "WH-001" not in engine.research_state.hypotheses

    def test_skipped_when_deps_unestablished(self):
        """VERIFIED WH with unestablished dep is NOT promoted."""
        from open_dirac.research_state import Hypothesis, HypothesisStatus, Verdict

        engine = self._make_engine()
        engine.research_state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
        )
        engine.research_state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            status=HypothesisStatus.WORKING,
            depends_on=["WH-001"],
            review=ReviewResult(verdict=Verdict.VERIFIED, summary="OK", iteration=4),
        )
        engine._auto_promote("WH-002")
        assert "WH-002" in engine.research_state.hypotheses
        assert "ER-002" not in engine.research_state.hypotheses

    def test_cascade_promotes_dependent(self):
        """Promoting WH-001 cascades to promote WH-002 that depends on it."""
        from open_dirac.research_state import Hypothesis, HypothesisStatus, Verdict

        engine = self._make_engine()
        engine.research_state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict=Verdict.VERIFIED, summary="OK", iteration=4),
        )
        engine.research_state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            status=HypothesisStatus.WORKING,
            depends_on=["WH-001"],
            review=ReviewResult(
                verdict=Verdict.VERIFIED, summary="OK too", iteration=4
            ),
        )
        engine._auto_promote("WH-001")
        # Both should be promoted
        assert "ER-001" in engine.research_state.hypotheses
        assert "ER-002" in engine.research_state.hypotheses
        assert "WH-001" not in engine.research_state.hypotheses
        assert "WH-002" not in engine.research_state.hypotheses

    def test_cascade_chain_three_deep(self):
        """Cascade works through a chain: WH-001 -> WH-002 -> WH-003."""
        from open_dirac.research_state import Hypothesis, HypothesisStatus, Verdict

        engine = self._make_engine()
        engine.research_state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict=Verdict.VERIFIED, summary="OK", iteration=4),
        )
        engine.research_state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            status=HypothesisStatus.WORKING,
            depends_on=["WH-001"],
            review=ReviewResult(verdict=Verdict.VERIFIED, summary="OK", iteration=4),
        )
        engine.research_state.hypotheses["WH-003"] = Hypothesis(
            id="WH-003",
            status=HypothesisStatus.WORKING,
            depends_on=["WH-002"],
            review=ReviewResult(verdict=Verdict.VERIFIED, summary="OK", iteration=4),
        )
        engine._auto_promote("WH-001")
        assert "ER-001" in engine.research_state.hypotheses
        assert "ER-002" in engine.research_state.hypotheses
        assert "ER-003" in engine.research_state.hypotheses

    def test_cascade_stops_at_unverified(self):
        """Cascade does not promote unverified WHs in the chain."""
        from open_dirac.research_state import Hypothesis, HypothesisStatus, Verdict

        engine = self._make_engine()
        engine.research_state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict=Verdict.VERIFIED, summary="OK", iteration=4),
        )
        engine.research_state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            status=HypothesisStatus.WORKING,
            depends_on=["WH-001"],
            # No review — not promotable
        )
        engine.research_state.hypotheses["WH-003"] = Hypothesis(
            id="WH-003",
            status=HypothesisStatus.WORKING,
            depends_on=["WH-002"],
            review=ReviewResult(verdict=Verdict.VERIFIED, summary="OK", iteration=4),
        )
        engine._auto_promote("WH-001")
        assert "ER-001" in engine.research_state.hypotheses
        assert "WH-002" in engine.research_state.hypotheses  # not promoted (no review)
        assert "WH-003" in engine.research_state.hypotheses  # blocked by WH-002


# ---------------------------------------------------------------------------
# Auto-expire critiques (Tier 1c)
# ---------------------------------------------------------------------------


class TestAutoExpireCritiques:
    """Test _auto_expire_critiques expiration logic."""

    def _make_engine(self, auto_expire_iterations=3):
        from open_dirac.engine import OpenDirac

        engine = OpenDirac.__new__(OpenDirac)
        engine.config = Config()
        engine.config.auto_expire_iterations = auto_expire_iterations
        engine.research_state = ResearchState()
        engine.workspace = MagicMock()
        engine.workspace.root = MagicMock()
        return engine

    def test_medium_critique_expires(self):
        """MEDIUM critique auto-expires after TTL iterations."""
        from open_dirac.research_state import Critique, CritiqueStatus, Severity

        engine = self._make_engine(auto_expire_iterations=3)
        engine.iteration = 8
        engine.research_state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.MEDIUM,
            status=CritiqueStatus.ACTIVE,
            argument="Minor.",
            iteration_filed=5,
        )
        engine._auto_expire_critiques()
        crit = engine.research_state.critiques["CRIT-001"]
        assert crit.status == CritiqueStatus.RESOLVED
        assert crit.resolution_type == "expired"
        assert crit.iteration_resolved == 8

    def test_low_critique_expires(self):
        """LOW critique auto-expires after TTL iterations."""
        from open_dirac.research_state import Critique, CritiqueStatus, Severity

        engine = self._make_engine(auto_expire_iterations=3)
        engine.iteration = 10
        engine.research_state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["STRATEGY"],
            severity=Severity.LOW,
            status=CritiqueStatus.ACTIVE,
            argument="Cosmetic.",
            iteration_filed=5,
        )
        engine._auto_expire_critiques()
        assert (
            engine.research_state.critiques["CRIT-001"].status
            == CritiqueStatus.RESOLVED
        )

    def test_high_critique_never_expires(self):
        """HIGH critique is never auto-expired regardless of age."""
        from open_dirac.research_state import Critique, CritiqueStatus, Severity

        engine = self._make_engine(auto_expire_iterations=3)
        engine.iteration = 100
        engine.research_state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.HIGH,
            status=CritiqueStatus.ACTIVE,
            argument="Critical flaw.",
            iteration_filed=5,
        )
        engine._auto_expire_critiques()
        assert (
            engine.research_state.critiques["CRIT-001"].status == CritiqueStatus.ACTIVE
        )

    def test_young_critique_not_expired(self):
        """MEDIUM critique younger than TTL is not expired."""
        from open_dirac.research_state import Critique, CritiqueStatus, Severity

        engine = self._make_engine(auto_expire_iterations=3)
        engine.iteration = 7
        engine.research_state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.MEDIUM,
            status=CritiqueStatus.ACTIVE,
            argument="Minor.",
            iteration_filed=5,
        )
        engine._auto_expire_critiques()
        assert (
            engine.research_state.critiques["CRIT-001"].status == CritiqueStatus.ACTIVE
        )

    def test_disabled_when_ttl_zero(self):
        """No expiry when auto_expire_iterations is 0."""
        from open_dirac.research_state import Critique, CritiqueStatus, Severity

        engine = self._make_engine(auto_expire_iterations=0)
        engine.iteration = 100
        engine.research_state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.MEDIUM,
            status=CritiqueStatus.ACTIVE,
            argument="Minor.",
            iteration_filed=5,
        )
        engine._auto_expire_critiques()
        assert (
            engine.research_state.critiques["CRIT-001"].status == CritiqueStatus.ACTIVE
        )

    def test_already_resolved_not_touched(self):
        """Already-resolved critiques are not re-expired."""
        from open_dirac.research_state import Critique, CritiqueStatus, Severity

        engine = self._make_engine(auto_expire_iterations=3)
        engine.iteration = 100
        engine.research_state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.MEDIUM,
            status=CritiqueStatus.RESOLVED,
            argument="Minor.",
            resolution_type="dismissed",
            resolution="Already handled.",
            iteration_filed=5,
            iteration_resolved=6,
        )
        engine._auto_expire_critiques()
        crit = engine.research_state.critiques["CRIT-001"]
        assert crit.resolution_type == "dismissed"  # unchanged
        assert crit.iteration_resolved == 6  # unchanged
