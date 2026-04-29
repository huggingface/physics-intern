"""Tests for the formatter agent and render_formatter_context renderer."""

from unittest.mock import MagicMock

import pytest

from open_dirac.agents.formatter import FormatterAgent
from open_dirac.agents.formatter.context import render_formatter_context
from open_dirac.llm import LLMResponse
from open_dirac.research_state import (
    Evidence,
    FailedApproach,
    Hypothesis,
    HypothesisStatus,
    ResearchState,
    ResearchQuestion,
    ReviewResult,
    RQStatus,
    Verdict,
)
from open_dirac.task import Task, TaskType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state_with_er_and_wh():
    """State with one ER, one WH, one open RQ, dead ends, and strategy."""
    state = ResearchState(
        problem_statement="Derive the Hawking temperature.",
        conventions="Natural units: G = c = hbar = k_B = 1.",
        strategy="Use Euclidean methods.",
        iteration=5,
    )
    state.hypotheses["ER-001"] = Hypothesis(
        id="ER-001",
        statement="T_H = 1/(8 pi M)",
        status=HypothesisStatus.ESTABLISHED,
        derivation="Surface gravity kappa = 1/(4M), then T = kappa/(2 pi).",
        evidence=[
            Evidence(
                type="compute",
                method="Symbolic computation",
                result="T = 1/(8*pi*M)",
                confidence="exact",
                iteration=3,
            )
        ],
        review=ReviewResult(
            verdict=Verdict.VERIFIED,
            summary="Confirmed.",
            iteration=4,
        ),
    )
    state.hypotheses["WH-002"] = Hypothesis(
        id="WH-002",
        statement="Entropy S = 4 pi M^2",
        status=HypothesisStatus.WORKING,
        derivation="Integration of dS = dM/T.",
        evidence=[
            Evidence(
                type="research",
                method="Analytical",
                result="S ~ 4*pi*M**2",
                confidence="approximate",
            )
        ],
    )
    state.hypotheses["WH-003"] = Hypothesis(
        id="WH-003",
        statement="Radiation is thermal",
        status=HypothesisStatus.ABANDONED,
    )
    state.research_questions["RQ-004"] = ResearchQuestion(
        id="RQ-004",
        question="What is the greybody factor?",
        status=RQStatus.OPEN,
    )
    state.failed_approaches.append(
        FailedApproach(
            description="Euclidean path integral",
            reason="Regularization issues",
        )
    )
    state.survey_background = "Some background."
    return state


@pytest.fixture
def empty_state():
    return ResearchState(problem_statement="Empty problem.", iteration=0)


# ===========================================================================
# render_formatter_context
# ===========================================================================


class TestRenderFormatterContext:
    def test_contains_problem_statement(self, state_with_er_and_wh):
        ctx = render_formatter_context(state_with_er_and_wh)
        assert "<problem-statement>" in ctx
        assert "Derive the Hawking temperature." in ctx
        assert "</problem-statement>" in ctx

    def test_contains_conventions(self, state_with_er_and_wh):
        ctx = render_formatter_context(state_with_er_and_wh)
        assert "<conventions>" in ctx
        assert "Natural units" in ctx
        assert "</conventions>" in ctx

    def test_no_conventions_tag_when_empty(self, empty_state):
        ctx = render_formatter_context(empty_state)
        assert "<conventions>" not in ctx

    def test_contains_established_results(self, state_with_er_and_wh):
        ctx = render_formatter_context(state_with_er_and_wh)
        assert "<established-results>" in ctx
        assert '<result id="ER-001">' in ctx
        assert "T_H = 1/(8 pi M)" in ctx
        assert "Surface gravity kappa" in ctx
        assert "Symbolic computation" in ctx
        assert "T = 1/(8*pi*M)" in ctx
        assert "exact" in ctx
        assert "VERIFIED" in ctx

    def test_excludes_working_hypotheses_from_results(self, state_with_er_and_wh):
        ctx = render_formatter_context(state_with_er_and_wh)
        # WH-002 should NOT appear inside <established-results>
        er_section_start = ctx.index("<established-results>")
        er_section_end = ctx.index("</established-results>")
        er_section = ctx[er_section_start:er_section_end]
        assert "WH-002" not in er_section
        assert "Entropy S = 4 pi M^2" not in er_section

    def test_excludes_strategy(self, state_with_er_and_wh):
        ctx = render_formatter_context(state_with_er_and_wh)
        assert "<strategy>" not in ctx
        assert "Euclidean methods" not in ctx

    def test_excludes_dead_ends(self, state_with_er_and_wh):
        ctx = render_formatter_context(state_with_er_and_wh)
        assert "<dead-ends>" not in ctx
        assert "Euclidean path integral" not in ctx

    def test_excludes_background_survey(self, state_with_er_and_wh):
        ctx = render_formatter_context(state_with_er_and_wh)
        assert "<background-survey>" not in ctx
        assert "Some background." not in ctx

    def test_excludes_abandoned_hypotheses(self, state_with_er_and_wh):
        ctx = render_formatter_context(state_with_er_and_wh)
        assert "WH-003" not in ctx
        assert "Radiation is thermal" not in ctx

    def test_unresolved_items_present(self, state_with_er_and_wh):
        ctx = render_formatter_context(state_with_er_and_wh)
        assert "<unresolved-items>" in ctx
        assert "RQ-004" in ctx
        assert "greybody factor" in ctx
        assert "WH-002" in ctx

    def test_no_unresolved_items_when_clean(self):
        state = ResearchState(problem_statement="Clean.")
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            statement="x = 1",
            status=HypothesisStatus.ESTABLISHED,
        )
        ctx = render_formatter_context(state)
        assert "<unresolved-items>" not in ctx

    def test_empty_established_results(self, empty_state):
        ctx = render_formatter_context(empty_state)
        assert "<established-results>" in ctx
        assert "No established results" in ctx

    def test_er_without_evidence_or_review(self):
        """ER with no evidence/review still renders statement."""
        state = ResearchState(problem_statement="Minimal.")
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            statement="Result A",
            status=HypothesisStatus.ESTABLISHED,
        )
        ctx = render_formatter_context(state)
        assert '<result id="ER-001">' in ctx
        assert "Result A" in ctx
        assert "<evidence>" not in ctx

    def test_multiple_ers_sorted(self):
        state = ResearchState(problem_statement="Multi.")
        for i in [3, 1, 2]:
            state.hypotheses[f"ER-{i:03d}"] = Hypothesis(
                id=f"ER-{i:03d}",
                statement=f"Result {i}",
                status=HypothesisStatus.ESTABLISHED,
            )
        ctx = render_formatter_context(state)
        pos1 = ctx.index("ER-001")
        pos2 = ctx.index("ER-002")
        pos3 = ctx.index("ER-003")
        assert pos1 < pos2 < pos3


# ===========================================================================
# FormatterAgent.build_context
# ===========================================================================


class TestFormatterBuildContext:
    def _make_agent(self, research_state, answer_template=""):
        config = MagicMock()
        workspace = MagicMock()
        metrics = MagicMock()
        agent = FormatterAgent(
            config, workspace, metrics, answer_template=answer_template
        )
        agent.research_state = research_state
        return agent

    def test_build_context_uses_render_formatter_context(self, state_with_er_and_wh):
        agent = self._make_agent(state_with_er_and_wh)
        task = Task(
            task_id="T-1",
            task_type=TaskType.FORMAT,
            assigned_to="formatter",
            priority="normal",
            iteration=5,
            body="Format.",
        )
        ctx = agent.build_context(task, iteration=5)
        assert "<problem-statement>" in ctx
        assert "<established-results>" in ctx
        # Should NOT contain old-style sections
        assert "# Strategy" not in ctx
        assert "# Dead Ends" not in ctx
        assert "<evidence-log>" not in ctx

    def test_build_context_includes_answer_template(self, state_with_er_and_wh):
        template = "T_H = FILL IN"
        agent = self._make_agent(state_with_er_and_wh, answer_template=template)
        task = Task(
            task_id="T-1",
            task_type=TaskType.FORMAT,
            assigned_to="formatter",
            priority="normal",
            iteration=5,
            body="Format.",
        )
        ctx = agent.build_context(task, iteration=5)
        assert "<answer-template>" in ctx
        assert "T_H = FILL IN" in ctx
        assert "</answer-template>" in ctx

    def test_build_context_no_answer_template(self, state_with_er_and_wh):
        agent = self._make_agent(state_with_er_and_wh, answer_template="")
        task = Task(
            task_id="T-1",
            task_type=TaskType.FORMAT,
            assigned_to="formatter",
            priority="normal",
            iteration=5,
            body="Format.",
        )
        ctx = agent.build_context(task, iteration=5)
        assert "<answer-template>" not in ctx


# ===========================================================================
# FormatterAgent.process_response — rejection detection
# ===========================================================================

SYMPY_TEMPLATE = """\
import sympy as sp

x, y = sp.symbols('x y')

def answer(x, y):
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    result = ...  # a SymPy expression of inputs
    choice = ...  # one of {'A', 'B', 'C', 'D'}
    # ---------------------------------------------------------------
    return result, choice
"""


class TestProcessResponseRejection:
    def _make_agent(self, answer_template=""):
        config = MagicMock()
        workspace = MagicMock()
        metrics = MagicMock()
        return FormatterAgent(
            config, workspace, metrics, answer_template=answer_template
        )

    def _make_response(self, text):
        return LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=1.0,
        )

    def _make_task(self):
        return Task(
            task_id="FORMAT-001",
            task_type=TaskType.FORMAT,
            assigned_to="formatter",
            iteration=1,
        )

    def test_llm_rejection_marker_sets_reason(self):
        agent = self._make_agent(answer_template=SYMPY_TEMPLATE)
        resp = self._make_response(
            "FORMATTER_REJECTION: Cannot fill Lambda placeholder\nDetails here."
        )
        agent.process_response(resp, self._make_task(), iteration=1)
        assert agent.rejection_reason == "Cannot fill Lambda placeholder"
        # Should still write the file (for circuit-breaker fallback)
        agent.workspace.write_file.assert_called_once()

    def test_good_output_no_rejection(self):
        agent = self._make_agent(answer_template=SYMPY_TEMPLATE)
        good_code = """\
import sympy as sp

x, y = sp.symbols('x y')

def answer(x, y):
    result = x**2 + y
    choice = 'A'
    return result, choice
"""
        resp = self._make_response(good_code)
        agent.process_response(resp, self._make_task(), iteration=1)
        assert agent.rejection_reason is None
        agent.workspace.write_file.assert_called_once()

    def test_no_template_always_passes(self):
        agent = self._make_agent(answer_template="")
        resp = self._make_response("anything goes")
        agent.process_response(resp, self._make_task(), iteration=1)
        assert agent.rejection_reason is None

    def test_rejection_reason_resets_between_calls(self):
        agent = self._make_agent(answer_template=SYMPY_TEMPLATE)
        task = self._make_task()
        # First call: rejected
        bad_resp = self._make_response("FORMATTER_REJECTION: missing data")
        agent.process_response(bad_resp, task, iteration=1)
        assert agent.rejection_reason is not None
        # Second call: accepted
        good_resp = self._make_response("""\
import sympy as sp

x, y = sp.symbols('x y')

def answer(x, y):
    result = x**2 + y
    choice = 'A'
    return result, choice
""")
        agent.process_response(good_resp, task, iteration=2)
        assert agent.rejection_reason is None
