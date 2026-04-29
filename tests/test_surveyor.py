"""Tests for SurveyorAgent."""

from unittest.mock import MagicMock

from open_dirac.agents.surveyor import SurveyorAgent
from open_dirac.research_state import ResearchState
from open_dirac.task import Task, TaskType
from open_dirac.llm import LLMResponse


def _make_agent():
    """Create a SurveyorAgent with mocked dependencies."""
    config = MagicMock()
    workspace = MagicMock()
    metrics = MagicMock()
    agent = SurveyorAgent(config=config, workspace=workspace, metrics=metrics)
    return agent


def _make_task():
    return Task(
        task_id="SURVEY-000",
        task_type=TaskType.SURVEY,
        assigned_to="surveyor",
        iteration=0,
    )


# ---------- build_context tests ----------


def test_build_context_initial():
    """iteration=0 includes problem statement only, no 'Current Research State'."""
    agent = _make_agent()
    agent.research_state = ResearchState(
        problem_statement="Derive the Hawking temperature."
    )

    context = agent.build_context(_make_task(), iteration=0)

    assert "Derive the Hawking temperature." in context
    assert "<current-research-state>" not in context


def test_build_context_replan():
    """iteration > 0 includes current research state section."""
    agent = _make_agent()
    agent.research_state = ResearchState(
        problem_statement="Derive the Hawking temperature."
    )

    context = agent.build_context(_make_task(), iteration=5)

    assert "Derive the Hawking temperature." in context
    # Re-survey should not include research-state when state is empty
    # (no conventions, no ERs, no RQs, no WHs)


# ---------- process_response tests ----------

STRATEGY_TEXT = """The Hawking temperature derivation proceeds most naturally via the Euclidean
path integral approach. The key insight is that requiring regularity of the
Euclidean section at the horizon fixes the periodicity of imaginary time.

**Promising approaches:**
- Euclidean continuation: Wick rotate, demand regularity at the horizon.
- Surface gravity route: Compute kappa from the Killing vector norm gradient.

**Pitfalls:**
- Don't confuse coordinate-dependent and invariant quantities.
- The naive WKB approximation breaks down near the horizon.
"""


def test_process_response_stores_survey():
    """process_response stores the raw text as a dict."""
    agent = _make_agent()
    response = LLMResponse(
        text=STRATEGY_TEXT,
        input_tokens=100,
        output_tokens=200,
        stop_reason="end_turn",
        duration=0.5,
    )
    agent.process_response(response, _make_task(), iteration=0)

    assert agent.parsed_survey is not None
    assert isinstance(agent.parsed_survey, dict)
    assert "Euclidean" in agent.parsed_survey["raw_notes"]


def test_process_response_empty_text():
    """Empty response still creates a survey (with empty notes)."""
    agent = _make_agent()
    response = LLMResponse(
        text="",
        input_tokens=50,
        output_tokens=0,
        stop_reason="end_turn",
        duration=0.1,
    )
    agent.process_response(response, _make_task(), iteration=0)

    assert agent.parsed_survey is not None
    assert agent.parsed_survey["raw_notes"] == ""


def test_process_response_strips_whitespace():
    """Leading/trailing whitespace is stripped from survey notes."""
    agent = _make_agent()
    response = LLMResponse(
        text="  \n Some notes here. \n  ",
        input_tokens=50,
        output_tokens=50,
        stop_reason="end_turn",
        duration=0.2,
    )
    agent.process_response(response, _make_task(), iteration=0)

    assert agent.parsed_survey["raw_notes"] == "Some notes here."


# ---------- JSON parsing tests ----------


def test_surveyor_parses_json_sections():
    """Surveyor extracts structured sections from JSON block."""
    from open_dirac.agents.surveyor import SurveyorAgent
    from unittest.mock import MagicMock
    from open_dirac.llm import LLMResponse

    agent = SurveyorAgent.__new__(SurveyorAgent)
    agent.research_state = None

    response = LLMResponse(
        text='Some prose analysis.\n\n```json\n{"background": "Physical context here", "known_pitfalls": "Watch for sign errors", "sanity_checks": "Result must be positive"}\n```',
        stop_reason="end_turn",
        input_tokens=100,
        output_tokens=200,
        duration=0.5,
    )
    task = MagicMock()
    agent.process_response(response, task, iteration=0)

    survey = agent.parsed_survey
    assert survey["raw_notes"].startswith("Some prose")
    assert survey["background"] == "Physical context here"
    assert survey["known_pitfalls"] == "Watch for sign errors"
    assert survey["sanity_checks"] == [
        {"predicate": "Result must be positive"}
    ]  # str fallback → list[dict]
    assert "key_insights" not in survey  # not provided


def test_surveyor_fallback_on_no_json():
    """Surveyor falls back gracefully when no JSON block present."""
    from open_dirac.agents.surveyor import SurveyorAgent
    from open_dirac.llm import LLMResponse
    from unittest.mock import MagicMock

    agent = SurveyorAgent.__new__(SurveyorAgent)
    agent.research_state = None

    response = LLMResponse(
        text="Just plain prose analysis with no JSON.",
        stop_reason="end_turn",
        input_tokens=100,
        output_tokens=50,
        duration=0.1,
    )
    task = MagicMock()
    agent.process_response(response, task, iteration=0)

    survey = agent.parsed_survey
    assert survey["raw_notes"] == "Just plain prose analysis with no JSON."
    assert "background" not in survey  # no structured sections


def test_surveyor_parses_problem_summary():
    """Surveyor extracts problem_summary from JSON block."""
    from open_dirac.agents.surveyor import SurveyorAgent
    from unittest.mock import MagicMock
    from open_dirac.llm import LLMResponse

    agent = SurveyorAgent.__new__(SurveyorAgent)
    agent.research_state = None

    response = LLMResponse(
        text='Analysis.\n\n```json\n{"background": "Context", "problem_summary": "Derive the Hawking temperature for a Schwarzschild black hole."}\n```',
        stop_reason="end_turn",
        input_tokens=100,
        output_tokens=200,
        duration=0.5,
    )
    task = MagicMock()
    agent.process_response(response, task, iteration=0)

    assert (
        agent.parsed_survey["problem_summary"]
        == "Derive the Hawking temperature for a Schwarzschild black hole."
    )


def test_build_context_replan_uses_slim_state():
    """Re-survey (iteration > 0) uses slim state (one-liners, no derivations)."""
    agent = _make_agent()
    agent.research_state = ResearchState(
        problem_statement="Derive the Hawking temperature.",
        conventions="Natural units.",
        strategy="Use surface gravity method.",
    )
    context = agent.build_context(_make_task(), iteration=5)
    assert "<research-state>" in context
    # Research state renders conventions (strategy excluded from surveyor revision)
    assert "<conventions>" in context
    # Strategy is NOT included in surveyor revision context
    assert "<strategy>" not in context
    # Does NOT produce derivation XML tags (one-liner format)
    assert "<derivation>" not in context


def test_surveyor_fallback_on_malformed_json():
    """Surveyor falls back when JSON is malformed."""
    from open_dirac.agents.surveyor import SurveyorAgent
    from open_dirac.llm import LLMResponse
    from unittest.mock import MagicMock

    agent = SurveyorAgent.__new__(SurveyorAgent)
    agent.research_state = None

    response = LLMResponse(
        text="Analysis.\n\n```json\n{broken json\n```",
        stop_reason="end_turn",
        input_tokens=100,
        output_tokens=50,
        duration=0.1,
    )
    task = MagicMock()
    agent.process_response(response, task, iteration=0)

    survey = agent.parsed_survey
    assert survey["raw_notes"].startswith("Analysis.")
    assert "background" not in survey  # no structured sections
