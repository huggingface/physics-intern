"""Tests for ResearcherAgent: JSON parsing, process_response, context building."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from open_dirac.agents.evidence_base import render_relevant_results
from open_dirac.agents.researcher import (
    ResearcherAgent,
    _extract_derivation_text,
    _parse_researcher_json,
)
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
# _parse_researcher_json
# ---------------------------------------------------------------------------


class TestParseResearcherJson:
    def test_fenced_json(self):
        text = (
            "Some derivation...\n"
            "```json\n"
            '{"result": "T_H = 1/(8*pi*M)", "method": "Euclidean", '
            '"confidence": "exact", "summary": "Hawking temperature derived"}\n'
            "```"
        )
        result = _parse_researcher_json(text)
        assert result is not None
        assert result["result"] == "T_H = 1/(8*pi*M)"
        assert result["method"] == "Euclidean"

    def test_bare_json_with_brace_counting(self):
        text = (
            "The derivation shows {that intermediate step} leads to:\n"
            '{"result": "S = k ln(W)", "method": "statistical mechanics", '
            '"confidence": "exact", "summary": "Entropy formula"}'
        )
        result = _parse_researcher_json(text)
        assert result is not None
        assert result["result"] == "S = k ln(W)"

    def test_last_fenced_wins(self):
        text = (
            '```json\n{"result": "wrong", "method": "a", "confidence": "partial", "summary": "x"}\n```\n'
            "More analysis...\n"
            '```json\n{"result": "correct", "method": "b", "confidence": "exact", "summary": "y"}\n```'
        )
        result = _parse_researcher_json(text)
        assert result is not None
        assert result["result"] == "correct"

    def test_no_json_returns_none(self):
        text = "Just plain text with no JSON."
        assert _parse_researcher_json(text) is None

    def test_invalid_json_returns_none(self):
        text = "```json\n{invalid json}\n```"
        assert _parse_researcher_json(text) is None

    def test_fenced_preferred_over_bare(self):
        text = (
            '{"result": "bare", "method": "a", "confidence": "exact", "summary": "x"}\n'
            '```json\n{"result": "fenced", "method": "b", "confidence": "exact", "summary": "y"}\n```'
        )
        result = _parse_researcher_json(text)
        assert result["result"] == "fenced"

    def test_json_without_result_key_skipped(self):
        """JSON blocks that don't contain 'result' key are ignored."""
        text = '```json\n{"verdict": "VERIFIED", "summary": "ok"}\n```'
        assert _parse_researcher_json(text) is None

    def test_nested_braces_in_reasoning(self):
        """Brace-counting handles nested braces in surrounding text."""
        text = (
            "The set {a, b, c} has 3 elements. "
            "The function f(x) = x^{2} gives:\n"
            '{"result": "f(2) = 4", "method": "direct computation", '
            '"confidence": "exact", "summary": "Squared"}'
        )
        result = _parse_researcher_json(text)
        assert result is not None
        assert result["result"] == "f(2) = 4"


# ---------------------------------------------------------------------------
# ResearcherAgent.process_response
# ---------------------------------------------------------------------------


def _make_researcher() -> ResearcherAgent:
    config = MagicMock()
    root = Path(tempfile.mkdtemp())
    workspace = MagicMock()
    workspace.root = root

    # Delegate write_file/read_file to actual filesystem
    def _write_file(relpath, content):
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    workspace.write_file = _write_file
    workspace.read_file = lambda relpath: (
        (root / relpath).read_text() if (root / relpath).exists() else ""
    )
    metrics = MagicMock()
    agent = ResearcherAgent(config, workspace, metrics)
    state = ResearchState(problem_statement="test")
    # Add an RQ and a WH for target tests
    state.research_questions["RQ-001"] = ResearchQuestion(
        id="RQ-001", question="What is the Hawking temperature?"
    )
    state.hypotheses["WH-002"] = Hypothesis(
        id="WH-002",
        statement="T_H = 1/(8*pi*M)",
        status=HypothesisStatus.WORKING,
    )
    agent.research_state = state
    return agent


class TestResearcherProcessResponse:
    def test_parses_json_and_builds_evidence(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive T_H",
            target_claim="RQ-001",
        )
        text = (
            "Starting from the Schwarzschild metric...\n\n"
            "```json\n"
            '{"result": "T_H = 1/(8*pi*M)", "method": "Euclidean path integral", '
            '"confidence": "exact", "summary": "Hawking temperature via Euclidean method"}\n'
            "```"
        )
        response = LLMResponse(
            text=text,
            input_tokens=500,
            output_tokens=200,
            stop_reason="end_turn",
            duration=1.0,
        )
        agent.process_response(response, task, iteration=1)
        assert len(agent.research_state.research_questions["RQ-001"].evidence) == 1
        ev = agent.research_state.research_questions["RQ-001"].evidence[0]
        assert ev.type == "research"
        assert ev.result == "T_H = 1/(8*pi*M)"
        assert ev.method == "Euclidean path integral"
        assert ev.confidence == "exact"
        assert ev.summary == "Hawking temperature via Euclidean method"
        # reasoning is the full response text
        assert "Schwarzschild metric" in ev.reasoning
        assert ev.iteration == 1

    def test_stores_on_wh_target(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Refine WH-002",
            target_claim="WH-002",
        )
        text = '```json\n{"result": "confirmed", "method": "direct", "confidence": "exact", "summary": "ok"}\n```'
        response = LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        agent.process_response(response, task, iteration=2)
        assert len(agent.research_state.hypotheses["WH-002"].evidence) == 1
        ev = agent.research_state.hypotheses["WH-002"].evidence[0]
        assert ev.result == "confirmed"
        assert ev.iteration == 2

    def test_fallback_on_parse_failure(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive something",
            target_claim="RQ-001",
        )
        response = LLMResponse(
            text="Long derivation without any JSON block at the end.",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        with pytest.raises(ParseFailureError):
            agent.process_response(response, task, iteration=3)
        assert len(agent.research_state.research_questions["RQ-001"].evidence) == 0

    def test_fallback_reasoning_truncated(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive something",
            target_claim="RQ-001",
        )
        long_text = "x" * 5000
        response = LLMResponse(
            text=long_text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        with pytest.raises(ParseFailureError):
            agent.process_response(response, task, iteration=1)
        assert len(agent.research_state.research_questions["RQ-001"].evidence) == 0

    def test_invalid_confidence_normalized(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive",
            target_claim="RQ-001",
        )
        text = '```json\n{"result": "ok", "method": "m", "confidence": "VERY_HIGH", "summary": "s"}\n```'
        response = LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        agent.process_response(response, task, iteration=1)
        assert len(agent.research_state.research_questions["RQ-001"].evidence) == 1
        ev = agent.research_state.research_questions["RQ-001"].evidence[0]
        assert ev.confidence == "partial"

    def test_target_extracted_from_body_when_no_target_claim(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Investigate RQ-001 further",
            target_claim="",
        )
        text = '```json\n{"result": "done", "method": "m", "confidence": "exact", "summary": "s"}\n```'
        response = LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        agent.process_response(response, task, iteration=1)
        assert len(agent.research_state.research_questions["RQ-001"].evidence) == 1
        ev = agent.research_state.research_questions["RQ-001"].evidence[0]
        assert ev.result == "done"

    def test_empty_response_text(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive",
            target_claim="RQ-001",
        )
        response = LLMResponse(
            text="",
            input_tokens=100,
            output_tokens=0,
            stop_reason="end_turn",
            duration=0.1,
        )
        with pytest.raises(ParseFailureError):
            agent.process_response(response, task, iteration=1)
        assert len(agent.research_state.research_questions["RQ-001"].evidence) == 0


# ---------------------------------------------------------------------------
# ResearcherAgent.build_context
# ---------------------------------------------------------------------------


class TestResearcherBuildContext:
    def test_includes_background(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive T_H",
            target_claim="RQ-001",
            background="Hawking radiation is thermal emission from black holes.",
        )
        ctx = agent.build_context(task, iteration=1)
        assert "<background>" in ctx
        assert "Hawking radiation" in ctx

    def test_includes_target(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive T_H",
            target_claim="RQ-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert "<target>" in ctx
        assert "Hawking temperature" in ctx

    def test_includes_method_hints(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive",
            target_claim="RQ-001",
            method_hints=["Use Euclidean path integral"],
        )
        ctx = agent.build_context(task, iteration=1)
        assert "Euclidean path integral" in ctx

    def test_includes_conventions(self):
        agent = _make_researcher()
        agent.research_state.conventions = "Natural units: ħ = c = k_B = 1"
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive",
            target_claim="RQ-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert "<conventions>" in ctx
        assert "Natural units" in ctx

    def test_includes_established_results(self):
        agent = _make_researcher()
        agent.research_state.hypotheses["ER-003"] = Hypothesis(
            id="ER-003",
            statement="Area law holds",
            status=HypothesisStatus.ESTABLISHED,
        )
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive",
            target_claim="RQ-001",
        )
        ctx = agent.build_context(task, iteration=1)
        assert "ER-003" in ctx
        assert "Area law holds" in ctx

    def test_no_research_state(self):
        agent = _make_researcher()
        agent.research_state = None
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive something",
        )
        ctx = agent.build_context(task, iteration=1)
        assert "<task>" in ctx

    def test_relevant_results_resolved_wh(self):
        """relevant_results WH IDs are resolved to statement + evidence summary."""
        agent = _make_researcher()
        agent.research_state.hypotheses["WH-002"].evidence = [
            Evidence(
                type="research",
                result="T_H = 1/(8*pi*M)",
                method="Euclidean",
                confidence="exact",
                summary="Hawking temperature via Euclidean method",
            )
        ]
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive",
            target_claim="RQ-001",
            relevant_results=["WH-002"],
        )
        ctx = agent.build_context(task, iteration=1)
        assert "<relevant-results>" in ctx
        assert "WH-002" in ctx
        assert "T_H = 1/(8*pi*M)" in ctx
        assert "Hawking temperature via Euclidean method" in ctx

    def test_relevant_results_resolved_rq(self):
        """relevant_results RQ IDs are resolved to question + evidence."""
        agent = _make_researcher()
        agent.research_state.research_questions["RQ-001"].evidence = [
            Evidence(
                type="compute",
                result="F(p) = 1 - 16/25 p^2",
                confidence="approximate",
                summary="Leading-order fidelity term",
            )
        ]
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive",
            target_claim="WH-002",
            relevant_results=["RQ-001"],
        )
        ctx = agent.build_context(task, iteration=1)
        assert "RQ-001" in ctx
        assert "Hawking temperature" in ctx  # the RQ question
        assert "Leading-order fidelity term" in ctx

    def test_relevant_results_unknown_id(self):
        """Unknown IDs render with (not found in current state)."""
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive",
            target_claim="RQ-001",
            relevant_results=["WH-999"],
        )
        ctx = agent.build_context(task, iteration=1)
        assert "WH-999" in ctx
        assert "not found" in ctx

    def test_relevant_results_free_text_passthrough(self):
        """Non-ID entries in relevant_results are passed through as-is."""
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive",
            target_claim="RQ-001",
            relevant_results=["The partition function diverges at T=0"],
        )
        ctx = agent.build_context(task, iteration=1)
        assert "partition function diverges" in ctx

    def test_relevant_results_no_research_state(self):
        """With no research_state, IDs are rendered as bare text."""
        agent = _make_researcher()
        agent.research_state = None
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive",
            relevant_results=["WH-002"],
        )
        ctx = agent.build_context(task, iteration=1)
        assert "WH-002" in ctx


# ---------------------------------------------------------------------------
# render_relevant_results (from evidence_base)
# ---------------------------------------------------------------------------


class TestRenderRelevantResults:
    def test_hypothesis_with_evidence(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="T_H = 1/(8*pi*M)",
            evidence=[
                Evidence(
                    type="research",
                    result="T_H derived",
                    confidence="exact",
                    summary="Hawking temperature",
                )
            ],
        )
        result = render_relevant_results(["WH-001"], state)
        assert "**WH-001**" in result
        assert "T_H = 1/(8*pi*M)" in result
        assert "Hawking temperature" in result
        assert "exact" in result

    def test_hypothesis_without_evidence(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="Some claim",
        )
        result = render_relevant_results(["WH-001"], state)
        assert "**WH-001**: Some claim" in result
        assert "Evidence" not in result

    def test_rq_resolved(self):
        state = ResearchState()
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="What is X?",
            evidence=[
                Evidence(
                    type="compute",
                    result="X = 42",
                    confidence="exact",
                    summary="Computed X",
                )
            ],
        )
        result = render_relevant_results(["RQ-001"], state)
        assert "**RQ-001**: What is X?" in result
        assert "Computed X" in result

    def test_mixed_ids_and_text(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="Claim A",
        )
        result = render_relevant_results(
            ["WH-001", "The theory predicts divergence"],
            state,
        )
        assert "**WH-001**: Claim A" in result
        assert "- The theory predicts divergence" in result

    def test_none_state(self):
        result = render_relevant_results(["WH-001"], None)
        assert "- WH-001" in result


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _extract_derivation_text
# ---------------------------------------------------------------------------


class TestExtractDerivationText:
    def test_extracts_text_before_json_fence(self):
        text = (
            "Starting from the metric...\n\n"
            "We derive T_H = 1/(8*pi*M).\n\n"
            '```json\n{"result": "T_H = 1/(8*pi*M)"}\n```'
        )
        result = _extract_derivation_text(text)
        assert "Starting from the metric" in result
        assert "We derive T_H" in result
        assert "```json" not in result

    def test_no_json_fence_returns_full_text(self):
        text = "Just a derivation with no JSON."
        assert _extract_derivation_text(text) == text

    def test_multiple_fences_uses_last(self):
        text = (
            "Part 1\n"
            '```json\n{"wrong": true}\n```\n'
            "Part 2\n"
            '```json\n{"result": "ok"}\n```'
        )
        result = _extract_derivation_text(text)
        assert "Part 1" in result
        assert "Part 2" in result


# ---------------------------------------------------------------------------
# Derivation file tests (process_response)
# ---------------------------------------------------------------------------


class TestResearcherDerivationFile:
    def test_derivation_file_written(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive T_H",
            target_claim="WH-002",
        )
        text = (
            "Starting from Schwarzschild metric...\n\n"
            "We find T_H = 1/(8*pi*M).\n\n"
            "```json\n"
            '{"result": "T_H = 1/(8*pi*M)", "method": "Euclidean", '
            '"confidence": "exact", "summary": "Hawking temperature"}\n'
            "```"
        )
        response = LLMResponse(
            text=text,
            input_tokens=500,
            output_tokens=200,
            stop_reason="end_turn",
            duration=1.0,
        )
        agent.process_response(response, task, iteration=3)
        # Check evidence has derivation_file set
        assert len(agent.research_state.hypotheses["WH-002"].evidence) == 1
        ev = agent.research_state.hypotheses["WH-002"].evidence[0]
        assert ev.derivation_file == "WH-002_003.md"
        # Check file was written with derivation content (no JSON block)
        content = agent.workspace.read_file("derivations/WH-002_003.md")
        assert "Schwarzschild metric" in content
        assert "```json" not in content

    def test_derivation_file_on_parse_failure(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive",
            target_claim="RQ-001",
        )
        response = LLMResponse(
            text="Long derivation without JSON.",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )
        with pytest.raises(ParseFailureError):
            agent.process_response(response, task, iteration=2)
        # Derivation file is written before parsing, but no evidence stored
        assert len(agent.research_state.research_questions["RQ-001"].evidence) == 0
        content = agent.workspace.read_file("derivations/RQ-001_002.md")
        assert "Long derivation" in content

    def test_empty_response_no_derivation_file(self):
        agent = _make_researcher()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Derive",
            target_claim="RQ-001",
        )
        response = LLMResponse(
            text="",
            input_tokens=100,
            output_tokens=0,
            stop_reason="end_turn",
            duration=0.1,
        )
        with pytest.raises(ParseFailureError):
            agent.process_response(response, task, iteration=1)
        assert len(agent.research_state.research_questions["RQ-001"].evidence) == 0

    def test_derivation_file_serialization_roundtrip(self):
        """derivation_file survives JSON serialization roundtrip."""
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="test",
            evidence=[
                Evidence(
                    type="research",
                    method="analytical",
                    result="ok",
                    confidence="exact",
                    derivation_file="WH-001_001.md",
                )
            ],
        )
        json_str = state.to_json()
        restored = ResearchState.from_json(json_str)
        assert len(restored.hypotheses["WH-001"].evidence) == 1
        ev = restored.hypotheses["WH-001"].evidence[0]
        assert ev.derivation_file == "WH-001_001.md"

    def test_derivation_file_serialization_rq(self):
        """derivation_file survives JSON roundtrip on RQ evidence."""
        state = ResearchState()
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="test",
            evidence=[
                Evidence(
                    type="research",
                    derivation_file="RQ-001_005.md",
                )
            ],
        )
        json_str = state.to_json()
        restored = ResearchState.from_json(json_str)
        assert len(restored.research_questions["RQ-001"].evidence) == 1
        ev = restored.research_questions["RQ-001"].evidence[0]
        assert ev.derivation_file == "RQ-001_005.md"


class TestResearcherConfig:
    def test_is_one_shot(self):
        assert ResearcherAgent.tools == []

    def test_name(self):
        assert ResearcherAgent.name == "researcher"

    def test_prompt_file(self):
        assert ResearcherAgent.prompt_file == "prompt.md"
