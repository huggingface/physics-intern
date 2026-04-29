"""Tests for ReviewerAgent: JSON parsing, context building, evidence filtering."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from open_dirac.agents.reviewer import ReviewerAgent, _parse_review_json
import pytest

from open_dirac.llm import LLMResponse, ParseFailureError
from open_dirac.research_state import (
    Evidence,
    Hypothesis,
    HypothesisStatus,
    ResearchQuestion,
    ResearchState,
)
from open_dirac.task import Task, TaskType


# ---------------------------------------------------------------------------
# _parse_review_json
# ---------------------------------------------------------------------------


class TestParseReviewJson:
    def test_fenced_json(self):
        text = 'Some analysis...\n```json\n{"verdict": "VERIFIED", "summary": "ok", "details": "all good"}\n```'
        result = _parse_review_json(text)
        assert result is not None
        assert result["verdict"] == "VERIFIED"
        assert result["summary"] == "ok"

    def test_bare_json(self):
        text = 'Analysis here.\n{"verdict": "REFUTED", "summary": "bug found", "details": "sign error"}'
        result = _parse_review_json(text)
        assert result is not None
        assert result["verdict"] == "REFUTED"

    def test_last_fenced_wins(self):
        text = (
            '```json\n{"verdict": "INCONCLUSIVE", "summary": "a", "details": "b"}\n```\n'
            "More analysis...\n"
            '```json\n{"verdict": "VERIFIED", "summary": "c", "details": "d"}\n```'
        )
        result = _parse_review_json(text)
        assert result is not None
        assert result["verdict"] == "VERIFIED"

    def test_no_json_returns_none(self):
        text = "Just plain text with no JSON."
        assert _parse_review_json(text) is None

    def test_invalid_json_returns_none(self):
        text = "```json\n{invalid json}\n```"
        assert _parse_review_json(text) is None

    def test_fenced_preferred_over_bare(self):
        text = (
            '{"verdict": "REFUTED", "summary": "x", "details": "y"}\n'
            '```json\n{"verdict": "VERIFIED", "summary": "a", "details": "b"}\n```'
        )
        result = _parse_review_json(text)
        assert result["verdict"] == "VERIFIED"


# ---------------------------------------------------------------------------
# ReviewerAgent.process_response
# ---------------------------------------------------------------------------


def _make_reviewer(workspace_root: Path) -> ReviewerAgent:
    config = MagicMock()
    workspace = MagicMock()
    workspace.root = workspace_root
    # read_file delegates to actual filesystem
    workspace.read_file = lambda relpath: (workspace_root / relpath).read_text()
    metrics = MagicMock()
    agent = ReviewerAgent(config, workspace, metrics)
    state = ResearchState(problem_statement="test")
    state.hypotheses["WH-001"] = Hypothesis(
        id="WH-001",
        statement="Hawking temperature is T_H = 1/(8*pi*M)",
        status=HypothesisStatus.WORKING,
    )
    agent.research_state = state
    return agent


class TestReviewerValidateResponse:
    def test_valid_response_passes(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        text = '```json\n{"verdict": "VERIFIED", "summary": "ok", "details": "d"}\n```'
        response = LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        assert agent._validate_response(response) is True

    def test_unparseable_response_fails(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        response = LLMResponse(
            text="No JSON here",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        assert agent._validate_response(response) is False

    def test_parse_retries_from_config(self):
        """parse_retries is now configured globally via Config.parse_retries."""
        from open_dirac.config import Config

        config = Config()
        assert config.parse_retries == 2


class TestReviewerProcessResponse:
    def test_parses_verified(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        text = 'Analysis...\n```json\n{"verdict": "VERIFIED", "summary": "Correct derivation.", "details": "All steps check out."}\n```'
        response = LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        agent.process_response(response, task, iteration=1)
        review = agent.research_state.hypotheses["WH-001"].review
        assert review is not None
        assert review.verdict == "VERIFIED"
        assert review.summary == "Correct derivation."
        assert review.iteration == 1

    def test_parses_refuted(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        text = '{"verdict": "REFUTED", "summary": "Sign error.", "details": "Line 3 has wrong sign."}'
        response = LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        agent.process_response(response, task, iteration=2)
        review = agent.research_state.hypotheses["WH-001"].review
        assert review.verdict == "REFUTED"

    def test_fallback_inconclusive_on_no_json(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        response = LLMResponse(
            text="Some analysis without JSON output.",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        with pytest.raises(ParseFailureError):
            agent.process_response(response, task, iteration=3)
        # No review stored on parse failure
        assert agent.research_state.hypotheses["WH-001"].review is None

    def test_invalid_verdict_normalized(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        text = '```json\n{"verdict": "MAYBE", "summary": "not sure", "details": "..."}\n```'
        response = LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        agent.process_response(response, task, iteration=1)
        review = agent.research_state.hypotheses["WH-001"].review
        assert review.verdict == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# ReviewerAgent.build_context — per-script computation blocks
# ---------------------------------------------------------------------------


class TestReviewerBuildContext:
    def test_computation_blocks_rendered(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        # Set up evidence with scripts
        comp_dir = root / "computations"
        comp_dir.mkdir()
        (comp_dir / "001_calc.py").write_text("import numpy as np\nprint(42)")
        (comp_dir / "001_calc.output").write_text("42")
        agent.research_state.hypotheses["WH-001"].evidence = [
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
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert '<computation name="001_calc.py">' in ctx
        assert "<purpose>Compute the answer</purpose>" in ctx
        assert "import numpy as np" in ctx
        assert "<output>\n42\n  </output>" in ctx

    def test_multiple_scripts(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        comp_dir = root / "computations"
        comp_dir.mkdir()
        (comp_dir / "001_setup.py").write_text("# setup")
        (comp_dir / "001_setup.output").write_text("ok")
        (comp_dir / "002_verify.py").write_text("# verify")
        (comp_dir / "002_verify.output").write_text("pass")
        agent.research_state.hypotheses["WH-001"].evidence = [
            Evidence(
                type="compute",
                scripts=["001_setup.py", "002_verify.py"],
                script_purposes={
                    "001_setup.py": "Setup data",
                    "002_verify.py": "Verify result",
                },
                method="numerical",
                result="ok",
                confidence="exact",
            )
        ]
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert '<computation name="001_setup.py">' in ctx
        assert '<computation name="002_verify.py">' in ctx
        assert "<purpose>Setup data</purpose>" in ctx
        assert "<purpose>Verify result</purpose>" in ctx

    def test_derivation_file_rendered(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        # Write derivation file
        deriv_dir = root / "derivations"
        deriv_dir.mkdir()
        (deriv_dir / "WH-001_001.md").write_text(
            "Starting from the Schwarzschild metric..."
        )
        agent.research_state.hypotheses["WH-001"].evidence = [
            Evidence(
                type="research",
                reasoning="full text including JSON block",
                derivation_file="WH-001_001.md",
                method="analytical",
                result="T_H = 1/(8*pi*M)",
                confidence="exact",
            )
        ]
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert '<derivation file="WH-001_001.md">' in ctx
        assert "Starting from the Schwarzschild metric" in ctx
        assert "<reasoning>" not in ctx

    def test_derivation_file_fallback_to_reasoning(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        # No derivation file on disk — should fall back to ev.reasoning
        agent.research_state.hypotheses["WH-001"].evidence = [
            Evidence(
                type="research",
                reasoning="Fallback reasoning text",
                derivation_file="WH-001_001.md",
                method="analytical",
                result="T_H = 1/(8*pi*M)",
                confidence="exact",
            )
        ]
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert '<derivation file="WH-001_001.md">' in ctx
        assert "Fallback reasoning text" in ctx

    def test_no_derivation_file_uses_reasoning(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        agent.research_state.hypotheses["WH-001"].evidence = [
            Evidence(
                type="research",
                reasoning="By direct derivation from the metric...",
                method="analytical",
                result="T_H = 1/(8*pi*M)",
                confidence="exact",
            )
        ]
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert "<reasoning>" in ctx
        assert "By direct derivation from the metric" in ctx

    def test_research_evidence_has_reasoning(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        agent.research_state.hypotheses["WH-001"].evidence = [
            Evidence(
                type="research",
                reasoning="By direct derivation from the metric...",
                method="analytical",
                result="T_H = 1/(8*pi*M)",
                confidence="exact",
            )
        ]
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert "<reasoning>" in ctx
        assert "By direct derivation from the metric" in ctx
        # No truncation
        assert "[truncated]" not in ctx

    def test_missing_script_shows_not_found(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        (root / "computations").mkdir()
        agent.research_state.hypotheses["WH-001"].evidence = [
            Evidence(
                type="compute",
                scripts=["missing_script.py"],
                method="numerical",
                result="42",
                confidence="exact",
            )
        ]
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert "[not found]" in ctx


# ---------------------------------------------------------------------------
# ReviewerAgent.build_context — problem statement & answer template
# ---------------------------------------------------------------------------


class TestReviewerProblemContext:
    def test_always_uses_full_problem_statement(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        agent.research_state.problem_summary = "Derive T_H for Schwarzschild BH."
        agent.research_state.problem_statement = "Full lengthy problem statement..."
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert "Full lengthy problem statement..." in ctx

    def test_uses_problem_statement_when_summary_empty(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        agent.research_state.problem_summary = ""
        agent.research_state.problem_statement = "Full lengthy problem statement..."
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert "Full lengthy problem statement..." in ctx

    def test_includes_answer_template(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        agent.research_state.problem_statement = "Test problem"
        agent.research_state.answer_template = "F(p) = <sympy expression>"
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert "<answer-template>" in ctx
        assert "F(p) = <sympy expression>" in ctx

    def test_no_answer_template_when_empty(self):
        root = Path(tempfile.mkdtemp())
        agent = _make_reviewer(root)
        agent.research_state.problem_statement = "Test problem"
        agent.research_state.answer_template = ""
        task = Task(
            task_id="T1",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Review WH-001",
            target_claim="WH-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert "<answer-template>" not in ctx


# ---------------------------------------------------------------------------
# Computer agent: evidence_scripts filtering + purposes
# ---------------------------------------------------------------------------


class TestComputerEvidenceFiltering:
    def _make_agent(self):
        from open_dirac.agents.computer import ComputerAgent

        agent = ComputerAgent.__new__(ComputerAgent)
        agent.research_state = ResearchState(problem_statement="test")
        rq_id = f"RQ-{agent.research_state.next_entity_num():03d}"
        agent.research_state.research_questions[rq_id] = ResearchQuestion(
            id=rq_id, question="Compute X?"
        )
        agent._last_script_names = ["001_setup.py", "002_main.py", "003_verify.py"]
        return agent, rq_id

    def _make_result(self, tool_calls):
        from open_dirac.llm import AgentResult

        return AgentResult(text="", tool_calls=tool_calls)

    def _make_tc(self, name, tool_input, output="ok", is_error=False):
        from open_dirac.tool_call import ToolCall

        return ToolCall(
            tool_name=name,
            tool_input=tool_input,
            output=output,
            is_error=is_error,
            duration=0.1,
        )

    def test_evidence_scripts_filters(self):
        agent, rq_id = self._make_agent()
        task = Task(
            task_id="T1",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            body=f"Compute {rq_id}",
            target_claim=rq_id,
        )
        tool_calls = [
            self._make_tc("document_approach", {"approach": "test"}),
            self._make_tc(
                "execute_python",
                {"purpose": "Setup", "code": "pass"},
                output="=== 001_setup.py ===\nPurpose: Setup\nExit: success\n\nok",
            ),
            self._make_tc(
                "execute_python",
                {"purpose": "Main calc", "code": "print(42)"},
                output="=== 002_main.py ===\nPurpose: Main calc\nExit: success\n\n42",
            ),
            self._make_tc(
                "execute_python",
                {"purpose": "Verify", "code": "print('pass')"},
                output="=== 003_verify.py ===\nPurpose: Verify\nExit: success\n\npass",
            ),
            self._make_tc(
                "submit_result",
                {
                    "target_id": rq_id,
                    "method": "numerical",
                    "result": "42",
                    "confidence": "exact",
                    "description": "d",
                    "notes": "n",
                    "evidence_scripts": ["002_main.py", "003_verify.py"],
                },
            ),
        ]
        result = self._make_result(tool_calls)
        agent.process_response(result, task, iteration=1)
        assert len(agent.research_state.research_questions[rq_id].evidence) == 1
        ev = agent.research_state.research_questions[rq_id].evidence[0]
        assert ev.scripts == ["002_main.py", "003_verify.py"]

    def test_no_evidence_scripts_uses_all(self):
        agent, rq_id = self._make_agent()
        task = Task(
            task_id="T1",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            body=f"Compute {rq_id}",
            target_claim=rq_id,
        )
        tool_calls = [
            self._make_tc("document_approach", {"approach": "test"}),
            self._make_tc("execute_python", {"purpose": "A", "code": "pass"}),
            self._make_tc("execute_python", {"purpose": "B", "code": "pass"}),
            self._make_tc("execute_python", {"purpose": "C", "code": "pass"}),
            self._make_tc(
                "submit_result",
                {
                    "target_id": rq_id,
                    "method": "m",
                    "result": "r",
                    "confidence": "exact",
                    "description": "d",
                    "notes": "n",
                },
            ),
        ]
        result = self._make_result(tool_calls)
        agent.process_response(result, task, iteration=1)
        ev = agent.research_state.research_questions[rq_id].evidence[-1]
        assert ev.scripts == ["001_setup.py", "002_main.py", "003_verify.py"]

    def test_invalid_evidence_scripts_falls_back(self):
        agent, rq_id = self._make_agent()
        task = Task(
            task_id="T1",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            body=f"Compute {rq_id}",
            target_claim=rq_id,
        )
        tool_calls = [
            self._make_tc("document_approach", {"approach": "test"}),
            self._make_tc("execute_python", {"purpose": "A", "code": "pass"}),
            self._make_tc(
                "submit_result",
                {
                    "target_id": rq_id,
                    "method": "m",
                    "result": "r",
                    "confidence": "exact",
                    "description": "d",
                    "notes": "n",
                    "evidence_scripts": ["nonexistent.py"],
                },
            ),
        ]
        result = self._make_result(tool_calls)
        agent.process_response(result, task, iteration=1)
        ev = agent.research_state.research_questions[rq_id].evidence[-1]
        # Falls back to all scripts
        assert ev.scripts == ["001_setup.py", "002_main.py", "003_verify.py"]

    def test_purposes_collected(self):
        agent, rq_id = self._make_agent()
        task = Task(
            task_id="T1",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            body=f"Compute {rq_id}",
            target_claim=rq_id,
        )
        tool_calls = [
            self._make_tc("document_approach", {"approach": "test"}),
            self._make_tc(
                "execute_python", {"purpose": "Initialize grid", "code": "pass"}
            ),
            self._make_tc(
                "execute_python", {"purpose": "Main calculation", "code": "pass"}
            ),
            self._make_tc(
                "execute_python", {"purpose": "Verify result", "code": "pass"}
            ),
            self._make_tc(
                "submit_result",
                {
                    "target_id": rq_id,
                    "method": "m",
                    "result": "r",
                    "confidence": "exact",
                    "description": "d",
                    "notes": "n",
                },
            ),
        ]
        result = self._make_result(tool_calls)
        agent.process_response(result, task, iteration=1)
        ev = agent.research_state.research_questions[rq_id].evidence[-1]
        assert ev.script_purposes == {
            "001_setup.py": "Initialize grid",
            "002_main.py": "Main calculation",
            "003_verify.py": "Verify result",
        }


# ---------------------------------------------------------------------------
# Evidence.script_purposes serialization
# ---------------------------------------------------------------------------


class TestEvidenceScriptPurposesSerialization:
    def test_roundtrip_via_json(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="test",
            evidence=[
                Evidence(
                    type="compute",
                    scripts=["001_calc.py"],
                    script_purposes={"001_calc.py": "Compute partition function"},
                    method="numerical",
                    result="42",
                    confidence="exact",
                )
            ],
        )
        json_str = state.to_json()
        restored = ResearchState.from_json(json_str)
        ev = restored.hypotheses["WH-001"].evidence[0]
        assert ev.script_purposes == {"001_calc.py": "Compute partition function"}

    def test_roundtrip_rq_evidence(self):
        state = ResearchState()
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="What is X?",
            evidence=[
                Evidence(
                    type="compute",
                    scripts=["001_calc.py"],
                    script_purposes={"001_calc.py": "Compute X"},
                    method="numerical",
                    result="42",
                    confidence="exact",
                )
            ],
        )
        json_str = state.to_json()
        restored = ResearchState.from_json(json_str)
        ev = restored.research_questions["RQ-001"].evidence[0]
        assert ev.script_purposes == {"001_calc.py": "Compute X"}

    def test_missing_script_purposes_defaults_empty(self):
        """Legacy JSON without script_purposes deserializes to empty dict."""
        import json

        data = {
            "hypotheses": {
                "WH-001": {
                    "id": "WH-001",
                    "evidence": [
                        {
                            "type": "compute",
                            "scripts": ["001_calc.py"],
                            "method": "numerical",
                            "result": "42",
                            "confidence": "exact",
                        }
                    ],
                }
            }
        }
        state = ResearchState.from_json(json.dumps(data))
        ev = state.hypotheses["WH-001"].evidence[0]
        assert ev.script_purposes == {}
