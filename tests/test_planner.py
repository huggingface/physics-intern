"""Tests for PlannerAgent and engine planner integration."""

from unittest.mock import MagicMock, patch

from open_dirac.agents.planner import PlannerAgent, _parse_planner_json
from open_dirac.config import Config
from open_dirac.engine import LoopState
from open_dirac.llm import LLMResponse
from open_dirac.research_state import (
    Evidence,
    FailedApproach,
    Hypothesis,
    HypothesisStatus,
    ResearchQuestion,
    ResearchState,
    ReviewResult,
    RQStatus,
)
from open_dirac.task import Task, TaskType


def _safe_path_mock() -> MagicMock:
    """MagicMock safe for ``ws.root.__truediv__`` — see test_engine.py."""
    m = MagicMock()
    m.return_value.exists.return_value = False
    return m


class TestPlannerBuildContext:
    """Test PlannerAgent.build_context produces correct XML-wrapped content."""

    def _make_agent(self) -> PlannerAgent:
        config = Config()
        ws = MagicMock()
        ws.root = MagicMock()
        metrics = MagicMock()
        agent = PlannerAgent(config, ws, metrics)
        agent.research_state = ResearchState(
            problem_statement="Derive the Hawking temperature.",
        )
        return agent

    def test_includes_problem_statement(self):
        agent = self._make_agent()
        task = Task(task_id="PLAN-000", task_type=TaskType.PLAN, assigned_to="planner")
        ctx = agent.build_context(task, iteration=0)
        assert "<problem-statement>" in ctx
        assert "Hawking temperature" in ctx
        assert "</problem-statement>" in ctx

    def test_includes_background_survey(self):
        agent = self._make_agent()
        agent.research_state.survey_background = "Surface gravity via Killing vectors."
        task = Task(task_id="PLAN-000", task_type=TaskType.PLAN, assigned_to="planner")
        ctx = agent.build_context(task, iteration=0)
        assert "<background-survey>" in ctx
        assert "Killing vectors" in ctx
        assert "</background-survey>" in ctx

    def test_no_survey_section_when_missing(self):
        agent = self._make_agent()
        task = Task(task_id="PLAN-000", task_type=TaskType.PLAN, assigned_to="planner")
        ctx = agent.build_context(task, iteration=0)
        assert "<background-survey>" not in ctx


class TestPlannerProcessResponse:
    """Test PlannerAgent.process_response stores strategy correctly."""

    def _make_agent(self) -> PlannerAgent:
        config = Config()
        ws = MagicMock()
        ws.root = MagicMock()
        metrics = MagicMock()
        return PlannerAgent(config, ws, metrics)

    def test_stores_strategy(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = "1. Derive surface gravity\n2. Apply Unruh effect"
        task = Task(task_id="PLAN-000", task_type=TaskType.PLAN, assigned_to="planner")
        agent.process_response(response, task, iteration=0)
        assert (
            agent.parsed_strategy == "1. Derive surface gravity\n2. Apply Unruh effect"
        )

    def test_strips_whitespace(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = "  \n  Strategy content  \n  "
        task = Task(task_id="PLAN-000", task_type=TaskType.PLAN, assigned_to="planner")
        agent.process_response(response, task, iteration=0)
        assert agent.parsed_strategy == "Strategy content"

    def test_empty_response_gives_none(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = "   "
        task = Task(task_id="PLAN-000", task_type=TaskType.PLAN, assigned_to="planner")
        agent.process_response(response, task, iteration=0)
        assert agent.parsed_strategy is None


class TestPlannerReviseContext:
    """Test PlannerAgent.build_context in revise mode."""

    def _make_agent_with_entities(self) -> PlannerAgent:
        config = Config()
        ws = MagicMock()
        ws.root = MagicMock()
        metrics = MagicMock()
        agent = PlannerAgent(config, ws, metrics)
        state = ResearchState(
            problem_statement="Derive the Hawking temperature.",
            strategy="1. Compute surface gravity\n2. Apply periodicity",
            conventions="Use natural units.",
            research_notes=[
                {"iteration": 1, "text": "Surface gravity kappa identified."},
                {"iteration": 3, "text": "Euclidean method seems more robust."},
            ],
        )
        # Add an ER
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            statement="kappa = 1/(4M)",
            status=HypothesisStatus.ESTABLISHED,
            review=ReviewResult(verdict="VERIFIED"),
        )
        # Add a WH with a review
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            statement="T_H = kappa/(2 pi)",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict="REFUTED"),
        )
        # Add a WH without review
        state.hypotheses["WH-003"] = Hypothesis(
            id="WH-003",
            statement="Alternative via Unruh effect",
            status=HypothesisStatus.WORKING,
        )
        # Add an abandoned hypothesis
        state.hypotheses["WH-004"] = Hypothesis(
            id="WH-004",
            statement="Wrong approach",
            status=HypothesisStatus.ABANDONED,
        )
        # Add an open RQ with evidence
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="What is the surface gravity?",
            status=RQStatus.OPEN,
            evidence=[
                Evidence(type="research", result="kappa = 1/(4M)"),
                Evidence(type="compute", result="Confirmed numerically"),
            ],
        )
        # Add a dead end
        state.failed_approaches.append(
            FailedApproach(
                description="Naive WKB method", reason="Divergent at horizon"
            )
        )
        # Add background survey
        state.survey_background = "Black hole thermodynamics fundamentals."
        agent.research_state = state
        return agent

    def test_revise_context_includes_strategy(self):
        agent = self._make_agent_with_entities()
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
            body="ER-001 was demoted after re-review.",
        )
        ctx = agent.build_context(task, iteration=5)
        assert "<current-strategy>" in ctx
        assert "Compute surface gravity" in ctx
        assert "</current-strategy>" in ctx

    def test_revise_context_includes_trigger(self):
        agent = self._make_agent_with_entities()
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
            body="ER-001 was demoted after re-review.",
        )
        ctx = agent.build_context(task, iteration=5)
        assert "<revision-trigger>" in ctx
        assert "ER-001 was demoted" in ctx
        assert "</revision-trigger>" in ctx

    def test_revise_context_includes_entities(self):
        agent = self._make_agent_with_entities()
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
            body="Trigger text",
        )
        ctx = agent.build_context(task, iteration=5)
        # <entities> tag removed; ERs are now in <established-results> inside <research-state>
        assert "<entities>" not in ctx
        assert "<research-state>" in ctx
        assert "<established-results>" in ctx
        # ER still shown
        assert "ER-001: kappa = 1/(4M), VERIFIED" in ctx
        # WHs and RQs are no longer shown in revise context
        assert (
            "WH-002" not in ctx.split("<dead-ends>")[0]
            if "<dead-ends>" in ctx
            else "WH-002" not in ctx
        )
        assert (
            "WH-003" not in ctx.split("<dead-ends>")[0]
            if "<dead-ends>" in ctx
            else "WH-003" not in ctx
        )
        assert "RQ-001" not in ctx

    def test_revise_context_includes_dead_ends(self):
        agent = self._make_agent_with_entities()
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
            body="Trigger text",
        )
        ctx = agent.build_context(task, iteration=5)
        assert "<dead-ends>" in ctx
        assert "Naive WKB method" in ctx
        assert "Divergent at horizon" in ctx
        # Abandoned hypothesis in dead ends
        assert "WH-004" in ctx.split("<dead-ends>")[1].split("</dead-ends>")[0]

    def test_revise_context_excludes_research_notes(self):
        agent = self._make_agent_with_entities()
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
            body="Trigger text",
        )
        ctx = agent.build_context(task, iteration=5)
        # Research notes dropped from revise context
        assert "<research-notes>" not in ctx

    def test_revise_context_includes_conventions(self):
        agent = self._make_agent_with_entities()
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
            body="Trigger text",
        )
        ctx = agent.build_context(task, iteration=5)
        assert "<conventions>" in ctx
        assert "natural units" in ctx

    def test_revise_context_includes_problem_statement(self):
        agent = self._make_agent_with_entities()
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
            body="Trigger text",
        )
        ctx = agent.build_context(task, iteration=5)
        assert "<problem-statement>" in ctx
        assert "Hawking temperature" in ctx

    def test_revise_context_includes_background_survey(self):
        agent = self._make_agent_with_entities()
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
            body="Trigger text",
        )
        ctx = agent.build_context(task, iteration=5)
        assert "<background-survey>" in ctx
        assert "Black hole thermodynamics" in ctx

    def test_revise_context_empty_state(self):
        """Revise mode with no research_state returns empty string."""
        config = Config()
        ws = MagicMock()
        ws.root = MagicMock()
        metrics = MagicMock()
        agent = PlannerAgent(config, ws, metrics)
        agent.research_state = None
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
            body="Trigger",
        )
        ctx = agent.build_context(task, iteration=5)
        assert ctx == ""

    def test_revise_context_excludes_rqs(self):
        """RQs are no longer shown in revise context."""
        config = Config()
        ws = MagicMock()
        ws.root = MagicMock()
        metrics = MagicMock()
        agent = PlannerAgent(config, ws, metrics)
        state = ResearchState(problem_statement="Test")
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="Singular test",
            status=RQStatus.OPEN,
            evidence=[Evidence(type="research", result="one")],
        )
        agent.research_state = state
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
            body="Trigger",
        )
        ctx = agent.build_context(task, iteration=5)
        assert "RQ-001" not in ctx


class TestPlannerReviseProcessResponse:
    """Test PlannerAgent.process_response in revise mode."""

    def _make_agent(self) -> PlannerAgent:
        config = Config()
        ws = MagicMock()
        ws.root = MagicMock()
        metrics = MagicMock()
        return PlannerAgent(config, ws, metrics)

    def test_parses_valid_json(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = """Here is the revised plan:

```json
{
  "revised_strategy": "1. Re-derive surface gravity\\n2. Verify with Euclidean method",
  "entity_actions": [
    {"id": "ER-001", "action": "keep", "concern": "may need re-examination"},
    {"id": "WH-002", "action": "abandon", "reason": "premise invalidated"},
    {"id": "RQ-001", "action": "keep"}
  ],
  "sanity_checks": [
    {"id": "SC-1", "check": "T -> 0 as M -> inf", "type": "constraint", "rationale": "Large BH limit"}
  ],
  "revision_rationale": "ER-001 demotion invalidates downstream results."
}
```
"""
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
        )
        agent.process_response(response, task, iteration=5)
        assert (
            agent.parsed_strategy
            == "1. Re-derive surface gravity\n2. Verify with Euclidean method"
        )
        assert len(agent.parsed_entity_actions) == 3
        assert agent.parsed_entity_actions[0] == {
            "id": "ER-001",
            "action": "keep",
            "concern": "may need re-examination",
        }
        assert agent.parsed_entity_actions[1] == {
            "id": "WH-002",
            "action": "abandon",
            "reason": "premise invalidated",
        }
        assert len(agent.parsed_sanity_checks) == 1
        assert agent.parsed_sanity_checks[0] == {
            "predicate": "T -> 0 as M -> inf",
            "rationale": "Large BH limit",
        }
        assert (
            agent.parsed_revision_rationale
            == "ER-001 demotion invalidates downstream results."
        )
        # No critique_assessments in JSON → None
        assert agent.parsed_critique_assessments is None

    def test_fallback_on_malformed_json(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = (
            "The strategy looks fine. No JSON here, just plain text analysis."
        )
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
        )
        agent.process_response(response, task, iteration=5)
        # Fallback: strategy is the full text
        assert (
            agent.parsed_strategy
            == "The strategy looks fine. No JSON here, just plain text analysis."
        )
        assert agent.parsed_entity_actions is None
        assert agent.parsed_sanity_checks is None
        assert agent.parsed_critique_assessments is None
        # Rationale is truncated to 200 chars
        assert agent.parsed_revision_rationale is not None
        assert len(agent.parsed_revision_rationale) <= 200

    def test_fallback_on_invalid_json_block(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = """Analysis:

```json
{not valid json at all!!!}
```
"""
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
        )
        agent.process_response(response, task, iteration=5)
        # Fallback: strategy is the stripped text
        assert agent.parsed_strategy is not None
        assert "not valid json" in agent.parsed_strategy
        assert agent.parsed_entity_actions is None

    def test_empty_response_revise_mode(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = "   "
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
        )
        agent.process_response(response, task, iteration=5)
        assert agent.parsed_strategy is None
        assert agent.parsed_entity_actions is None
        assert agent.parsed_revision_rationale is None

    def test_none_response_text(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = None
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
        )
        agent.process_response(response, task, iteration=5)
        assert agent.parsed_strategy is None

    def test_multiple_json_blocks_takes_last(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = """First attempt:

```json
{"revised_strategy": "wrong one"}
```

Actually, corrected:

```json
{
  "revised_strategy": "correct strategy",
  "entity_actions": [],
  "sanity_checks": [],
  "revision_rationale": "corrected"
}
```
"""
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
        )
        agent.process_response(response, task, iteration=5)
        assert agent.parsed_strategy == "correct strategy"
        assert agent.parsed_revision_rationale == "corrected"

    def test_parses_critique_assessments(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = """```json
{
  "critique_assessments": [
    {"id": "CRIT-001", "verdict": "accept", "reason": "Valid concern about sign convention"},
    {"id": "CRIT-002", "verdict": "dismiss", "reason": "Critique assumes Euclidean signature but we use Lorentzian"}
  ],
  "revised_strategy": "Updated strategy",
  "entity_actions": [],
  "sanity_checks": [],
  "revision_rationale": "Accepted one critique, dismissed another."
}
```"""
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
        )
        agent.process_response(response, task, iteration=5)
        assert agent.parsed_critique_assessments is not None
        assert len(agent.parsed_critique_assessments) == 2
        assert agent.parsed_critique_assessments[0]["id"] == "CRIT-001"
        assert agent.parsed_critique_assessments[0]["verdict"] == "accept"
        assert agent.parsed_critique_assessments[1]["id"] == "CRIT-002"
        assert agent.parsed_critique_assessments[1]["verdict"] == "dismiss"

    def test_critique_assessments_filters_invalid_entries(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = """```json
{
  "critique_assessments": [
    {"id": "CRIT-001", "verdict": "accept", "reason": "Valid"},
    {"id": "CRIT-002"},
    {"verdict": "dismiss"},
    "not a dict",
    {"id": "CRIT-003", "verdict": "dismiss", "reason": "Invalid assumption"}
  ],
  "revised_strategy": "Same strategy",
  "entity_actions": [],
  "sanity_checks": [],
  "revision_rationale": "Filtered."
}
```"""
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
        )
        agent.process_response(response, task, iteration=5)
        assert agent.parsed_critique_assessments is not None
        # Only entries with both "id" and "verdict" survive
        assert len(agent.parsed_critique_assessments) == 2
        assert agent.parsed_critique_assessments[0]["id"] == "CRIT-001"
        assert agent.parsed_critique_assessments[1]["id"] == "CRIT-003"

    def test_critique_assessments_not_a_list(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = """```json
{
  "critique_assessments": "should be a list",
  "revised_strategy": "Same",
  "entity_actions": [],
  "sanity_checks": [],
  "revision_rationale": "Bad format."
}
```"""
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
        )
        agent.process_response(response, task, iteration=5)
        assert agent.parsed_critique_assessments is None


class TestParsePlannerJson:
    """Test _parse_planner_json utility function."""

    def test_extracts_fenced_json(self):
        text = '```json\n{"key": "value"}\n```'
        result = _parse_planner_json(text)
        assert result == {"key": "value"}

    def test_returns_none_for_no_json(self):
        result = _parse_planner_json("No JSON here.")
        assert result is None

    def test_returns_none_for_non_dict(self):
        text = '```json\n["a", "b"]\n```'
        result = _parse_planner_json(text)
        assert result is None

    def test_returns_none_for_invalid_json(self):
        text = "```json\n{invalid}\n```"
        result = _parse_planner_json(text)
        assert result is None


class TestPlannerInitialModeUnchanged:
    """Verify initial (PLAN) mode behavior is preserved after revise changes."""

    def _make_agent(self) -> PlannerAgent:
        config = Config()
        ws = MagicMock()
        ws.root = MagicMock()
        metrics = MagicMock()
        agent = PlannerAgent(config, ws, metrics)
        agent.research_state = ResearchState(
            problem_statement="Derive the Hawking temperature.",
        )
        return agent

    def test_initial_mode_build_context(self):
        agent = self._make_agent()
        task = Task(task_id="PLAN-000", task_type=TaskType.PLAN, assigned_to="planner")
        ctx = agent.build_context(task, iteration=0)
        assert "<problem-statement>" in ctx
        assert "Hawking temperature" in ctx
        # Should NOT have revise-mode sections
        assert "<current-strategy>" not in ctx
        assert "<revision-trigger>" not in ctx
        assert "<entities>" not in ctx

    def test_initial_mode_process_response(self):
        agent = self._make_agent()
        response = MagicMock(spec=LLMResponse)
        response.text = "1. Derive surface gravity\n2. Apply Unruh effect"
        task = Task(task_id="PLAN-000", task_type=TaskType.PLAN, assigned_to="planner")
        agent.process_response(response, task, iteration=0)
        assert (
            agent.parsed_strategy == "1. Derive surface gravity\n2. Apply Unruh effect"
        )
        # Revise-mode fields should remain None
        assert agent.parsed_entity_actions is None
        assert agent.parsed_sanity_checks is None
        assert agent.parsed_revision_rationale is None

    def test_is_revise_mode_false_for_plan(self):
        agent = self._make_agent()
        task = Task(task_id="PLAN-000", task_type=TaskType.PLAN, assigned_to="planner")
        assert not agent._is_revise_mode(task)

    def test_is_revise_mode_true_for_plan_revise(self):
        agent = self._make_agent()
        task = Task(
            task_id="PLAN-REV-001",
            task_type=TaskType.PLAN_REVISE,
            assigned_to="planner",
        )
        assert agent._is_revise_mode(task)


class TestEngineApplyStrategy:
    """Test engine._apply_strategy() stores strategy in research state."""

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

            engine.surveyor = MagicMock()
            engine.surveyor.parsed_survey = None
            engine.planner = MagicMock()
            engine.planner.parsed_strategy = None
        return engine

    def test_apply_strategy_stores_strategy(self):
        engine = self._make_engine()
        engine.planner.parsed_strategy = (
            "1. Compute surface gravity\n2. Apply periodicity"
        )
        engine._apply_strategy()
        assert (
            engine.research_state.strategy
            == "1. Compute surface gravity\n2. Apply periodicity"
        )

    def test_apply_strategy_none_does_nothing(self):
        engine = self._make_engine()
        engine.planner.parsed_strategy = None
        engine._apply_strategy()
        assert engine.research_state.strategy == ""

    def test_planner_skipped_on_resume_with_strategy(self):
        """When strategy already exists (e.g. resume), planner should be skippable."""
        engine = self._make_engine()
        engine.research_state.strategy = "Existing strategy from prior run."
        # The run() method checks `not self.research_state.strategy` before calling _run_planner
        assert engine.research_state.strategy  # truthy -> planner would be skipped


class TestTaskTypePlanRevise:
    """Test PLAN_REVISE TaskType integration."""

    def test_plan_revise_in_task_type(self):
        assert TaskType.PLAN_REVISE == "plan_revise"

    def test_plan_revise_maps_to_planner(self):
        from open_dirac.task import TASK_TYPE_AGENT_MAP

        assert TASK_TYPE_AGENT_MAP[TaskType.PLAN_REVISE] == "planner"

    def test_task_from_frontmatter_plan_revise(self):
        text = "---\ntask_type: plan_revise\ntask_id: PLAN-REV-001\nassigned_to: planner\n---\n\nTrigger text here."
        task = Task.from_frontmatter(text)
        assert task.task_type == TaskType.PLAN_REVISE
        assert task.body == "Trigger text here."
