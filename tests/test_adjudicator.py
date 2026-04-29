"""Tests for AdjudicatorAgent: JSON parsing, context building, fallback."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from open_dirac.agents.adjudicator import AdjudicatorAgent, _parse_adjudication_json
from open_dirac.llm import LLMResponse
from open_dirac.research_state import (
    Evidence,
    Hypothesis,
    HypothesisStatus,
    ResearchState,
    ReviewResult,
)
from open_dirac.task import Task, TaskType


# ---------------------------------------------------------------------------
# _parse_adjudication_json
# ---------------------------------------------------------------------------


class TestParseAdjudicationJson:
    def test_fenced_json_valid(self):
        text = (
            "Analysis...\n```json\n"
            '{"adjudication": "valid", "reasoning": "Sign error found.", "revised_verdict": "REFUTED"}\n'
            "```"
        )
        result = _parse_adjudication_json(text)
        assert result is not None
        assert result["adjudication"] == "valid"
        assert result["reasoning"] == "Sign error found."

    def test_fenced_json_invalid(self):
        text = (
            "Analysis...\n```json\n"
            '{"adjudication": "invalid", "reasoning": "Critique wrong.", "counter_argument": "Sign is correct."}\n'
            "```"
        )
        result = _parse_adjudication_json(text)
        assert result is not None
        assert result["adjudication"] == "invalid"
        assert result["counter_argument"] == "Sign is correct."

    def test_fenced_json_needs_evidence(self):
        text = (
            "```json\n"
            '{"adjudication": "needs_evidence", "reasoning": "Unclear.", "investigation_scope": "Check limit."}\n'
            "```"
        )
        result = _parse_adjudication_json(text)
        assert result is not None
        assert result["adjudication"] == "needs_evidence"
        assert result["investigation_scope"] == "Check limit."

    def test_bare_json(self):
        text = 'Analysis here.\n{"adjudication": "valid", "reasoning": "error", "revised_verdict": "REFUTED"}'
        result = _parse_adjudication_json(text)
        assert result is not None
        assert result["adjudication"] == "valid"

    def test_last_fenced_wins(self):
        text = (
            '```json\n{"adjudication": "invalid", "reasoning": "a", "counter_argument": "b"}\n```\n'
            "More analysis...\n"
            '```json\n{"adjudication": "valid", "reasoning": "c", "revised_verdict": "REFUTED"}\n```'
        )
        result = _parse_adjudication_json(text)
        assert result is not None
        assert result["adjudication"] == "valid"

    def test_no_json_returns_none(self):
        text = "Just plain text with no JSON."
        assert _parse_adjudication_json(text) is None

    def test_invalid_json_returns_none(self):
        text = "```json\n{invalid json}\n```"
        assert _parse_adjudication_json(text) is None

    def test_fenced_preferred_over_bare(self):
        text = (
            '{"adjudication": "valid", "reasoning": "x", "revised_verdict": "REFUTED"}\n'
            '```json\n{"adjudication": "invalid", "reasoning": "a", "counter_argument": "b"}\n```'
        )
        result = _parse_adjudication_json(text)
        assert result["adjudication"] == "invalid"


# ---------------------------------------------------------------------------
# AdjudicatorAgent.process_response
# ---------------------------------------------------------------------------


def _make_adjudicator(workspace_root: Path) -> AdjudicatorAgent:
    config = MagicMock()
    workspace = MagicMock()
    workspace.root = workspace_root
    workspace.read_file = lambda relpath: (workspace_root / relpath).read_text()
    metrics = MagicMock()
    agent = AdjudicatorAgent(config, workspace, metrics)
    state = ResearchState(problem_statement="Derive the Hawking temperature.")
    state.hypotheses["ER-001"] = Hypothesis(
        id="ER-001",
        statement="T_H = 1/(8*pi*M)",
        status=HypothesisStatus.ESTABLISHED,
        evidence=[
            Evidence(
                type="research",
                method="analytical",
                result="T_H = 1/(8*pi*M)",
                confidence="exact",
            )
        ],
        review=ReviewResult(verdict="VERIFIED", summary="Correct.", iteration=1),
    )
    agent.research_state = state
    return agent


class TestAdjudicatorProcessResponse:
    def test_parses_valid(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Sign error in derivation.",
        )
        text = (
            "Analysis...\n```json\n"
            '{"adjudication": "valid", "reasoning": "Sign error confirmed.", "revised_verdict": "REFUTED"}\n'
            "```"
        )
        response = LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        agent.process_response(response, task, iteration=2)
        result = agent.adjudication_result
        assert result is not None
        assert result["adjudication"] == "valid"
        assert result["reasoning"] == "Sign error confirmed."
        assert result["revised_verdict"] == "REFUTED"

    def test_parses_invalid(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Wrong sign.",
        )
        text = (
            "```json\n"
            '{"adjudication": "invalid", "reasoning": "Sign is actually correct.", '
            '"counter_argument": "The critic misread the convention."}\n'
            "```"
        )
        response = LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        agent.process_response(response, task, iteration=2)
        result = agent.adjudication_result
        assert result["adjudication"] == "invalid"
        assert result["counter_argument"] == "The critic misread the convention."

    def test_parses_needs_evidence(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Unclear derivation.",
        )
        text = (
            "```json\n"
            '{"adjudication": "needs_evidence", "reasoning": "Cannot resolve.", '
            '"investigation_scope": "Recheck limiting case M->inf."}\n'
            "```"
        )
        response = LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        agent.process_response(response, task, iteration=2)
        result = agent.adjudication_result
        assert result["adjudication"] == "needs_evidence"
        assert result["investigation_scope"] == "Recheck limiting case M->inf."

    def test_fallback_on_no_json(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Wrong sign.",
        )
        response = LLMResponse(
            text="Some analysis without JSON output.",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        agent.process_response(response, task, iteration=3)
        result = agent.adjudication_result
        assert result["adjudication"] == "needs_evidence"
        assert "Failed to parse" in result["reasoning"]

    def test_invalid_adjudication_value_normalized(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Wrong sign.",
        )
        text = '```json\n{"adjudication": "MAYBE", "reasoning": "not sure"}\n```'
        response = LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        agent.process_response(response, task, iteration=1)
        result = agent.adjudication_result
        assert result["adjudication"] == "needs_evidence"


# ---------------------------------------------------------------------------
# AdjudicatorAgent.build_context
# ---------------------------------------------------------------------------


class TestAdjudicatorBuildContext:
    def test_includes_claim_and_challenge(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="The sign is wrong in step 3.",
        )
        ctx = agent.build_context(task, iteration=2)
        assert "<problem-statement>" in ctx
        assert "Derive the Hawking temperature" in ctx
        assert '<claim id="ER-001">' in ctx
        assert "T_H = 1/(8*pi*M)" in ctx
        assert "<challenge>" in ctx
        assert "The sign is wrong in step 3." in ctx

    def test_includes_conventions(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        agent.research_state.conventions = "G = c = hbar = 1 (natural units)"
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Wrong.",
        )
        ctx = agent.build_context(task, iteration=2)
        assert "<conventions>" in ctx
        assert "G = c = hbar = 1" in ctx

    def test_excludes_target_from_established_context(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        # Add another ER
        agent.research_state.hypotheses["ER-002"] = Hypothesis(
            id="ER-002",
            statement="Entropy S = A/(4*G)",
            status=HypothesisStatus.ESTABLISHED,
        )
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Wrong.",
        )
        ctx = agent.build_context(task, iteration=2)
        assert "<established-results>" in ctx
        assert "ER-002" in ctx
        # ER-001 should NOT appear in established-results (it's the challenged claim)
        # It does appear in <claim>, but not in established-results
        ec_start = ctx.index("<established-results>")
        ec_end = ctx.index("</established-results>")
        ec_section = ctx[ec_start:ec_end]
        assert "ER-001" not in ec_section

    def test_no_established_context_when_only_target(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Wrong.",
        )
        ctx = agent.build_context(task, iteration=2)
        # Only ER is the target itself, so no established-results section
        assert "<established-results>" not in ctx

    def test_includes_review_info(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Wrong.",
        )
        ctx = agent.build_context(task, iteration=2)
        assert "Original review verdict: VERIFIED" in ctx
        assert "Original review summary: Correct." in ctx

    def test_includes_computation_scripts(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        comp_dir = root / "computations"
        comp_dir.mkdir()
        (comp_dir / "001_calc.py").write_text("import numpy as np\nprint(42)")
        (comp_dir / "001_calc.output").write_text("42")
        agent.research_state.hypotheses["ER-001"].evidence = [
            Evidence(
                type="compute",
                approach="Direct calculation",
                scripts=["001_calc.py"],
                script_purposes={"001_calc.py": "Compute the answer"},
                output="42",
                method="numerical",
                result="42",
                confidence="exact",
            )
        ]
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Calculation is wrong.",
        )
        ctx = agent.build_context(task, iteration=2)
        assert '<computation name="001_calc.py">' in ctx
        assert "<purpose>Compute the answer</purpose>" in ctx
        assert "import numpy as np" in ctx

    def test_includes_sanity_checks_from_planner(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        agent.research_state.sanity_checks = [
            {
                "id": "SC-1",
                "check": "In the M -> inf limit, T_H -> 0",
                "type": "constraint",
                "rationale": "Known limit",
            }
        ]
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Wrong.",
        )
        ctx = agent.build_context(task, iteration=2)
        assert "<sanity-checks>" in ctx
        assert "M -> inf limit" in ctx

    def test_includes_known_pitfalls(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        agent.research_state.known_pitfalls = (
            "Coordinate singularity confusion at the horizon"
        )
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Wrong.",
        )
        ctx = agent.build_context(task, iteration=2)
        assert "<known-pitfalls>" in ctx
        assert "Coordinate singularity confusion" in ctx

    def test_no_known_pitfalls_when_empty(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Wrong.",
        )
        ctx = agent.build_context(task, iteration=2)
        assert "<known-pitfalls>" not in ctx

    def test_empty_research_state(self):
        root = Path(tempfile.mkdtemp())
        config = MagicMock()
        workspace = MagicMock()
        workspace.root = root
        metrics = MagicMock()
        agent = AdjudicatorAgent(config, workspace, metrics)
        # research_state is None
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="ER-001",
            critique_argument="Wrong.",
        )
        ctx = agent.build_context(task, iteration=1)
        assert ctx == ""

    def test_missing_target_claim(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_adjudicator(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.ADJUDICATE,
            assigned_to="adjudicator",
            target_claim="WH-999",
            critique_argument="Wrong.",
        )
        ctx = agent.build_context(task, iteration=2)
        # Should still include problem statement and challenge
        assert "<problem-statement>" in ctx
        assert "<challenge>" in ctx
        # But no claim-under-review (target not found)
        assert "<claim-under-review" not in ctx
