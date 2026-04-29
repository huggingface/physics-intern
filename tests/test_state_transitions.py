"""Tests for state_transitions: demote_hypothesis, promote_hypothesis, normalize_references."""

from open_dirac.research_state import (
    Hypothesis,
    HypothesisStatus,
    ResearchQuestion,
    ResearchState,
)
from open_dirac.state_transitions import (
    demote_hypothesis,
    normalize_references,
    promote_hypothesis,
)


class TestNormalizeReferences:
    def test_updates_stale_depends_on_wh_to_er(self):
        """depends_on referencing WH-002 should be updated when hypothesis is ER-002."""
        state = ResearchState()
        state.hypotheses["ER-002"] = Hypothesis(
            id="ER-002", status=HypothesisStatus.ESTABLISHED
        )
        state.hypotheses["WH-003"] = Hypothesis(
            id="WH-003",
            depends_on=["WH-002"],
        )
        normalize_references(state)
        assert state.hypotheses["WH-003"].depends_on == ["ER-002"]

    def test_updates_stale_depends_on_er_to_wh(self):
        """depends_on referencing ER-001 should be updated when hypothesis was demoted to WH-001."""
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001", status=HypothesisStatus.WORKING
        )
        state.hypotheses["WH-003"] = Hypothesis(
            id="WH-003",
            depends_on=["ER-001"],
        )
        normalize_references(state)
        assert state.hypotheses["WH-003"].depends_on == ["WH-001"]

    def test_idempotent(self):
        """Calling normalize_references twice produces the same result."""
        state = ResearchState()
        state.hypotheses["ER-002"] = Hypothesis(
            id="ER-002", status=HypothesisStatus.ESTABLISHED
        )
        state.hypotheses["WH-003"] = Hypothesis(
            id="WH-003",
            depends_on=["WH-002"],
        )
        normalize_references(state)
        deps_after_first = list(state.hypotheses["WH-003"].depends_on)
        normalize_references(state)
        assert state.hypotheses["WH-003"].depends_on == deps_after_first

    def test_no_alias_match_preserves_dep(self):
        """depends_on targeting a number that doesn't exist in hypotheses is left alone."""
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            depends_on=["WH-099"],
        )
        normalize_references(state)
        assert state.hypotheses["WH-002"].depends_on == ["WH-099"]


class TestNormalizeReferencesDependsOn:
    """Tests for normalize_references remapping depends_on entries."""

    def test_depends_on_remapped_on_promotion(self):
        """When WH-001 is promoted to ER-001, depends_on entries are remapped."""
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            depends_on=["WH-001"],
        )
        normalize_references(state)
        assert state.hypotheses["WH-002"].depends_on == ["ER-001"]

    def test_depends_on_unchanged_when_no_rename(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(id="WH-001")
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            depends_on=["WH-001"],
        )
        normalize_references(state)
        assert state.hypotheses["WH-002"].depends_on == ["WH-001"]


class TestNormalizeReferencesResolvedTo:
    def test_resolved_to_remapped(self):
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="test",
            resolved_to=["WH-001"],
        )
        normalize_references(state)
        assert state.research_questions["RQ-001"].resolved_to == ["ER-001"]


class TestDemoteHypothesis:
    def test_demotes_er_to_wh(self):
        state = ResearchState()
        state.hypotheses["ER-002"] = Hypothesis(
            id="ER-002",
            statement="Energy conserved",
            status=HypothesisStatus.ESTABLISHED,
        )
        new_id = demote_hypothesis(state, "ER-002")
        assert new_id == "WH-002"
        assert "ER-002" not in state.hypotheses
        assert "WH-002" in state.hypotheses
        assert state.hypotheses["WH-002"].status == HypothesisStatus.WORKING
        assert state.hypotheses["WH-002"].statement == "Energy conserved"

    def test_fixes_depends_on_references(self):
        state = ResearchState()
        state.hypotheses["ER-002"] = Hypothesis(
            id="ER-002",
            status=HypothesisStatus.ESTABLISHED,
        )
        state.hypotheses["WH-003"] = Hypothesis(
            id="WH-003",
            depends_on=["ER-002"],
        )
        demote_hypothesis(state, "ER-002")
        assert state.hypotheses["WH-003"].depends_on == ["WH-002"]

    def test_returns_none_for_wh(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
        )
        assert demote_hypothesis(state, "WH-001") is None
        # WH-001 should be unchanged
        assert "WH-001" in state.hypotheses
        assert state.hypotheses["WH-001"].status == HypothesisStatus.WORKING

    def test_returns_none_for_missing(self):
        state = ResearchState()
        assert demote_hypothesis(state, "ER-999") is None

    def test_preserves_other_hypotheses(self):
        state = ResearchState()
        state.hypotheses["ER-002"] = Hypothesis(
            id="ER-002",
            status=HypothesisStatus.ESTABLISHED,
        )
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
        )
        demote_hypothesis(state, "ER-002")
        assert "WH-001" in state.hypotheses
        assert state.hypotheses["WH-001"].status == HypothesisStatus.WORKING


class TestPromoteHypothesis:
    def test_promotes_wh_to_er(self):
        state = ResearchState()
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            statement="Energy conserved",
            status=HypothesisStatus.WORKING,
        )
        new_id = promote_hypothesis(state, "WH-002", iteration=7)
        assert new_id == "ER-002"
        assert "WH-002" not in state.hypotheses
        assert "ER-002" in state.hypotheses
        assert state.hypotheses["ER-002"].status == HypothesisStatus.ESTABLISHED
        assert state.hypotheses["ER-002"].statement == "Energy conserved"

    def test_stamps_iteration_modified(self):
        state = ResearchState()
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            status=HypothesisStatus.WORKING,
            iteration_modified=1,
        )
        promote_hypothesis(state, "WH-002", iteration=42)
        assert state.hypotheses["ER-002"].iteration_modified == 42

    def test_returns_none_for_er(self):
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        assert promote_hypothesis(state, "ER-001", iteration=1) is None
        assert "ER-001" in state.hypotheses
        assert state.hypotheses["ER-001"].status == HypothesisStatus.ESTABLISHED

    def test_returns_none_for_missing(self):
        state = ResearchState()
        assert promote_hypothesis(state, "WH-999", iteration=1) is None

    def test_fixes_depends_on_references(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
        )
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            status=HypothesisStatus.WORKING,
            depends_on=["WH-001"],
        )
        promote_hypothesis(state, "WH-001", iteration=5)
        assert state.hypotheses["WH-002"].depends_on == ["ER-001"]
