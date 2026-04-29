"""Tests for validation.py — post-integration validation and termination gates."""

from open_dirac.validation import (
    Violation,
    ViolationSeverity,
    validate_post_integration,
    can_terminate,
    check_er_demotion_safety,
    check_phantom_labels,
    check_stale_unverified_labels,
    check_critique_resolution_consistency,
)
from open_dirac.research_state import (
    Critique,
    CritiqueStatus,
    Evidence,
    Hypothesis,
    HypothesisStatus,
    ResearchQuestion,
    ResearchState,
    RQStatus,
    Severity,
    Verdict,
    ReviewResult,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockMetrics:
    """Minimal mock for MetricsTracker with last_critic_iteration."""

    def __init__(self, last_critic_iteration: int = 0):
        self.last_critic_iteration = last_critic_iteration


class MockConfig:
    """Minimal mock for Config."""

    def __init__(self):
        self.min_er_for_completion = 1


class MockWorkspace:
    """Minimal workspace mock for can_terminate (still takes workspace param)."""

    def __init__(self):
        self.root = None


# ---------------------------------------------------------------------------
# Violation dataclass tests
# ---------------------------------------------------------------------------


class TestViolation:
    def test_creation(self):
        v = Violation(
            check="test_check",
            severity=ViolationSeverity.ERROR,
            message="test msg",
        )
        assert v.check == "test_check"
        assert v.severity == ViolationSeverity.ERROR
        assert v.message == "test msg"
        assert v.detail == ""

    def test_with_detail(self):
        v = Violation(
            check="x",
            severity=ViolationSeverity.WARNING,
            message="y",
            detail="d",
        )
        assert v.detail == "d"


# ---------------------------------------------------------------------------
# check_er_demotion_safety
# ---------------------------------------------------------------------------


class TestErDemotionSafety:
    def test_demotes_er_with_refuted_verification(self):
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            statement="T = 1/(8piM)",
            status=HypothesisStatus.ESTABLISHED,
            review=ReviewResult(
                verdict=Verdict.REFUTED,
                summary="Calculation error",
                iteration=1,
            ),
        )
        violations = check_er_demotion_safety(state)
        assert len(violations) == 1
        assert "demoted" in violations[0].message
        # ER-001 should be demoted to WH-001
        assert "WH-001" in state.hypotheses
        assert "ER-001" not in state.hypotheses

    def test_no_demotion_when_verified(self):
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            statement="T = 1/(8piM)",
            status=HypothesisStatus.ESTABLISHED,
            review=ReviewResult(
                verdict=Verdict.VERIFIED,
                summary="Confirmed",
                iteration=2,
            ),
        )
        violations = check_er_demotion_safety(state)
        assert len(violations) == 0
        assert "ER-001" in state.hypotheses

    def test_no_demotion_when_no_verification(self):
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        violations = check_er_demotion_safety(state)
        assert len(violations) == 0

    def test_no_demotion_for_wh(self):
        """WH hypotheses are not demoted — only ER."""
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(
                verdict=Verdict.REFUTED,
                summary="Failed",
                iteration=1,
            ),
        )
        violations = check_er_demotion_safety(state)
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# check_phantom_labels
# ---------------------------------------------------------------------------


class TestPhantomLabels:
    def test_strips_unsubstantiated_verified(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            derivation="WH-001 is VERIFIED by computation.",
        )
        violations = check_phantom_labels(state)
        assert len(violations) == 1
        assert "[unverified]" in state.hypotheses["WH-001"].derivation

    def test_keeps_backed_verified(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            derivation="WH-001 is VERIFIED by computation.",
            review=ReviewResult(
                verdict=Verdict.VERIFIED,
                summary="Confirmed",
                iteration=1,
            ),
        )
        violations = check_phantom_labels(state)
        assert len(violations) == 0
        assert "VERIFIED" in state.hypotheses["WH-001"].derivation

    def test_no_action_without_verified(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            derivation="Some derivation text.",
        )
        violations = check_phantom_labels(state)
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# check_stale_unverified_labels
# ---------------------------------------------------------------------------


class TestStaleUnverifiedLabels:
    def test_promotes_unverified_to_verified(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            derivation="WH-001 is [unverified] pending verification.",
            review=ReviewResult(
                verdict=Verdict.VERIFIED,
                summary="Confirmed",
                iteration=1,
            ),
        )
        violations = check_stale_unverified_labels(state)
        # Note: the function returns [] (empty list) but mutates derivation
        assert "VERIFIED" in state.hypotheses["WH-001"].derivation
        assert "[unverified]" not in state.hypotheses["WH-001"].derivation

    def test_no_change_without_verified_verification(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            derivation="WH-001 is [unverified] pending verification.",
        )
        check_stale_unverified_labels(state)
        assert "[unverified]" in state.hypotheses["WH-001"].derivation


# ---------------------------------------------------------------------------
# check_critique_resolution_consistency
# ---------------------------------------------------------------------------


class TestCritiqueResolutionConsistency:
    def test_flags_vanished_target(self):
        state = ResearchState()
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-099"],
            severity=Severity.HIGH,
            status=CritiqueStatus.RESOLVED,
        )
        violations = check_critique_resolution_consistency(state)
        assert len(violations) == 1
        assert "no longer exists" in violations[0].message

    def test_no_flag_when_target_exists(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.HIGH,
            status=CritiqueStatus.RESOLVED,
        )
        violations = check_critique_resolution_consistency(state)
        assert len(violations) == 0

    def test_flags_label_coexistence(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.HIGH,
            argument="Inconsistent label: WH-001 should be ER-001",
            status=CritiqueStatus.RESOLVED,
        )
        violations = check_critique_resolution_consistency(state)
        assert any("co-exist" in v.message for v in violations)

    def test_skips_active_critiques(self):
        state = ResearchState()
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-099"],
            severity=Severity.HIGH,
            status=CritiqueStatus.ACTIVE,
        )
        violations = check_critique_resolution_consistency(state)
        assert len(violations) == 0

    def test_target_found_via_promoted_form(self):
        """If critique targets WH-001 but it was promoted to ER-001, no flag."""
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["WH-001"],
            severity=Severity.HIGH,
            status=CritiqueStatus.RESOLVED,
        )
        violations = check_critique_resolution_consistency(state)
        vanished = [v for v in violations if "no longer exists" in v.message]
        assert len(vanished) == 0

    def test_resolved_strategy_critique_no_false_positive(self):
        """Resolved critique targeting STRATEGY should not generate a vanished-target violation."""
        state = ResearchState()
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["STRATEGY"],
            severity=Severity.MEDIUM,
            argument="Strategy recommends refuted approach.",
            status=CritiqueStatus.RESOLVED,
            resolution="Updated strategy section.",
        )
        violations = check_critique_resolution_consistency(state)
        assert len(violations) == 0


class TestStrategyTerminationBlocking:
    """HIGH STRATEGY critiques no longer block termination (gate removed)."""

    def test_high_strategy_critique_does_not_block_termination(self):
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
            review=ReviewResult(
                verdict=Verdict.VERIFIED,
                summary="Confirmed",
                iteration=1,
            ),
        )
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=["STRATEGY"],
            severity=Severity.HIGH,
            status=CritiqueStatus.ACTIVE,
            argument="Strategy recommends a refuted approach.",
        )
        allowed, blockers = can_terminate(
            MockWorkspace(),
            MockConfig(),
            MockMetrics(last_critic_iteration=1),
            research_state=state,
        )
        assert allowed
        assert blockers == []


# ---------------------------------------------------------------------------
# validate_post_integration
# ---------------------------------------------------------------------------


class TestValidatePostIntegration:
    def test_returns_empty_for_clean_state(self):
        state = ResearchState()
        violations = validate_post_integration(state)
        assert violations == []

    def test_aggregates_violations(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            derivation="WH-001 is VERIFIED.",
        )
        violations = validate_post_integration(state)
        assert len(violations) >= 1
        assert any(v.check == "phantom_labels" for v in violations)

    def test_runs_four_checks(self):
        """Pipeline has exactly 4 checks."""
        from open_dirac.validation import _DEFAULT_CHECKS

        assert len(_DEFAULT_CHECKS) == 4


# ---------------------------------------------------------------------------
# can_terminate
# ---------------------------------------------------------------------------


class TestCanTerminate:
    def _make_state(self, **kwargs) -> ResearchState:
        return ResearchState(**kwargs)

    def test_allows_with_er_and_critic(self):
        state = self._make_state()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
            review=ReviewResult(
                verdict=Verdict.VERIFIED,
                summary="Confirmed",
                iteration=1,
            ),
        )
        allowed, blockers = can_terminate(
            MockWorkspace(),
            MockConfig(),
            MockMetrics(last_critic_iteration=1),
            research_state=state,
        )
        assert allowed
        assert blockers == []

    def test_blocks_without_critic(self):
        state = self._make_state()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
            review=ReviewResult(
                verdict=Verdict.VERIFIED,
                summary="Confirmed",
                iteration=1,
            ),
        )
        allowed, blockers = can_terminate(
            MockWorkspace(),
            MockConfig(),
            MockMetrics(last_critic_iteration=0),
            research_state=state,
        )
        assert not allowed
        assert any("critic" in b.lower() for b in blockers)

    def test_high_critiques_do_not_block_termination(self):
        """HIGH critiques no longer block termination (gate removed)."""
        state = self._make_state()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            severity=Severity.HIGH,
            status=CritiqueStatus.ACTIVE,
        )
        allowed, blockers = can_terminate(
            MockWorkspace(),
            MockConfig(),
            MockMetrics(last_critic_iteration=1),
            research_state=state,
        )
        assert allowed
        assert blockers == []

    def test_blocks_with_open_rq(self):
        state = self._make_state()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="Open question",
            status=RQStatus.OPEN,
        )
        allowed, blockers = can_terminate(
            MockWorkspace(),
            MockConfig(),
            MockMetrics(last_critic_iteration=1),
            research_state=state,
        )
        assert not allowed
        assert any("RQ-001" in b for b in blockers)

    def test_blocks_with_working_hypothesis(self):
        state = self._make_state()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            status=HypothesisStatus.WORKING,
        )
        allowed, blockers = can_terminate(
            MockWorkspace(),
            MockConfig(),
            MockMetrics(last_critic_iteration=1),
            research_state=state,
        )
        assert not allowed
        assert any("WH-002" in b for b in blockers)

    def test_blocks_wh_with_verified_but_unestablished_deps(self):
        """WH with VERIFIED review + unestablished deps → blocker about deps."""
        state = self._make_state()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
            depends_on=["WH-002"],
            review=ReviewResult(
                verdict=Verdict.VERIFIED,
                summary="Confirmed",
                iteration=1,
            ),
        )
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            status=HypothesisStatus.WORKING,
        )
        allowed, blockers = can_terminate(
            MockWorkspace(),
            MockConfig(),
            MockMetrics(last_critic_iteration=1),
            research_state=state,
        )
        assert not allowed
        wh1_blocker = [b for b in blockers if "WH-001" in b][0]
        assert "unestablished" in wh1_blocker.lower()
        assert "promote_hypothesis" not in wh1_blocker

    def test_blocks_wh_with_verified_no_deps_unexpected(self):
        """WH with VERIFIED review + no dep issues → unexpected blocker."""
        state = self._make_state()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(
                verdict=Verdict.VERIFIED,
                summary="Confirmed",
                iteration=1,
            ),
        )
        allowed, blockers = can_terminate(
            MockWorkspace(),
            MockConfig(),
            MockMetrics(last_critic_iteration=1),
            research_state=state,
        )
        assert not allowed
        wh1_blocker = [b for b in blockers if "WH-001" in b][0]
        assert "unexpected" in wh1_blocker.lower()
        assert "promote_hypothesis" not in wh1_blocker

    def test_wh_without_verification_gets_review_message(self):
        """WH without review result should say 'emit review', not 'promote'."""
        state = self._make_state()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
            evidence=[
                Evidence(type="research", reasoning="Some analysis", iteration=1)
            ],
        )
        allowed, blockers = can_terminate(
            MockWorkspace(),
            MockConfig(),
            MockMetrics(last_critic_iteration=1),
            research_state=state,
        )
        assert not allowed
        # Should say "review", not "promote"
        wh_blocker = [b for b in blockers if "WH-001" in b][0]
        assert "review" in wh_blocker.lower()
        assert "promote_hypothesis" not in wh_blocker
