"""Tests for deep critic one-shot JSON parsing and process_response."""

import tempfile
from unittest.mock import MagicMock


from open_dirac.agents.critic import CriticAgent, _parse_critic_json
from open_dirac.research_state import (
    Critique,
    CritiqueStatus,
    Hypothesis,
    HypothesisStatus,
    ResearchState,
    Severity,
)


# ---------------------------------------------------------------------------
# _parse_critic_json tests
# ---------------------------------------------------------------------------


class TestParseCriticJson:
    def test_fenced_json(self):
        text = 'Some analysis.\n```json\n{"summary": "ok", "details": "d", "critiques": []}\n```\n'
        result = _parse_critic_json(text)
        assert result is not None
        assert result["summary"] == "ok"
        assert result["critiques"] == []

    def test_fenced_json_last_match(self):
        """Takes the last fenced JSON block."""
        text = (
            '```json\n{"summary": "first", "details": "d", "critiques": []}\n```\n'
            "More text.\n"
            '```json\n{"summary": "second", "details": "d", "critiques": [{"severity": "HIGH", "target_id": "WH-001", "argument": "bad"}]}\n```\n'
        )
        result = _parse_critic_json(text)
        assert result is not None
        assert result["summary"] == "second"
        assert len(result["critiques"]) == 1

    def test_bare_json_with_nested_braces(self):
        text = (
            "My analysis is complete.\n"
            '{"summary": "done", "details": "full analysis", "critiques": '
            '[{"severity": "MEDIUM", "target_id": "ER-001", "argument": "issue"}]}'
        )
        result = _parse_critic_json(text)
        assert result is not None
        assert result["summary"] == "done"
        assert len(result["critiques"]) == 1
        assert result["critiques"][0]["severity"] == "MEDIUM"

    def test_clean_review_empty_array(self):
        text = '```json\n{"summary": "all clear", "details": "no issues", "critiques": []}\n```'
        result = _parse_critic_json(text)
        assert result is not None
        assert result["critiques"] == []

    def test_free_text_before_json(self):
        text = (
            "## Strategy Assessment\n"
            "The strategy looks sound.\n\n"
            "## Result Coherence\n"
            "All results are consistent.\n\n"
            '```json\n{"summary": "reviewed strategy and coherence", "details": "all good", "critiques": []}\n```'
        )
        result = _parse_critic_json(text)
        assert result is not None
        assert "strategy" in result["summary"]

    def test_parse_failure_returns_none(self):
        text = "This is just free text with no JSON at all."
        assert _parse_critic_json(text) is None

    def test_malformed_json_returns_none(self):
        text = '```json\n{"summary": "broken, "critiques": []}\n```'
        assert _parse_critic_json(text) is None

    def test_fenced_without_critiques_key_returns_none(self):
        """JSON block without 'critiques' key is skipped."""
        text = '```json\n{"verdict": "VERIFIED", "summary": "ok"}\n```'
        assert _parse_critic_json(text) is None

    def test_bare_json_without_critiques_key_skipped(self):
        text = '{"verdict": "VERIFIED", "summary": "ok"}'
        assert _parse_critic_json(text) is None

    def test_multiple_critiques(self):
        text = '```json\n{"summary": "s", "details": "d", "critiques": [{"severity": "HIGH", "target_id": "WH-001", "argument": "a"}, {"severity": "LOW", "target_id": "STRATEGY", "argument": "b"}]}\n```'
        result = _parse_critic_json(text)
        assert result is not None
        assert len(result["critiques"]) == 2


# ---------------------------------------------------------------------------
# process_response tests
# ---------------------------------------------------------------------------


def _make_state(**kwargs) -> ResearchState:
    return ResearchState(**kwargs)


def _make_agent() -> CriticAgent:
    config = MagicMock()
    workspace = MagicMock()
    workspace.root = tempfile.mkdtemp()
    metrics = MagicMock()
    agent = CriticAgent(config, workspace, metrics)
    return agent


def _make_response(text: str):
    resp = MagicMock()
    resp.text = text
    return resp


class TestCriticProcessResponse:
    def test_critiques_numbered_from_state(self):
        """CRIT-NNN numbering uses research_state.next_critique_num()."""
        agent = _make_agent()
        state = _make_state()
        # Pre-populate one critique so next is CRIT-002
        state.critiques["CRIT-001"] = Critique(
            id="CRIT-001",
            targets=[],
            severity=Severity.LOW,
            argument="old",
            status=CritiqueStatus.ACTIVE,
            iteration_filed=1,
        )
        agent.research_state = state
        task = MagicMock()
        resp = _make_response(
            '```json\n{"summary": "s", "details": "d", "critiques": '
            '[{"severity": "HIGH", "target_id": "WH-001", "argument": "new issue"}]}\n```'
        )
        agent.process_response(resp, task, iteration=3)
        assert not agent._no_critiques_filed
        assert "CRIT-002" in state.critiques
        crit = state.critiques["CRIT-002"]
        assert crit.severity == Severity.HIGH
        assert crit.iteration_filed == 3

    def test_no_critiques_filed_flag(self):
        """Empty critiques array sets _no_critiques_filed = True."""
        agent = _make_agent()
        agent.research_state = _make_state()
        task = MagicMock()
        resp = _make_response(
            '```json\n{"summary": "clean", "details": "d", "critiques": []}\n```'
        )
        agent.process_response(resp, task, iteration=2)
        assert agent._no_critiques_filed
        assert len(agent.research_state.critic_clean_reviews) == 1
        assert agent.research_state.critic_clean_reviews[0]["summary"] == "clean"

    def test_hypothesis_linking(self):
        """Filed critiques are linked to target hypotheses."""
        agent = _make_agent()
        state = _make_state()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="test",
            status=HypothesisStatus.WORKING,
            iteration_created=1,
        )
        agent.research_state = state
        task = MagicMock()
        resp = _make_response(
            '```json\n{"summary": "s", "details": "d", "critiques": '
            '[{"severity": "MEDIUM", "target_id": "WH-001", "argument": "issue"}]}\n```'
        )
        agent.process_response(resp, task, iteration=2)
        assert len(state.hypotheses["WH-001"].critiques) == 1

    def test_parse_failure_fallback(self):
        """Unparseable response → clean review with parse failure note."""
        agent = _make_agent()
        agent.research_state = _make_state()
        task = MagicMock()
        resp = _make_response("Just some text, no JSON.")
        agent.process_response(resp, task, iteration=1)
        assert agent._no_critiques_filed
        assert len(agent.research_state.critic_clean_reviews) == 1
        assert (
            "Parse failure" in agent.research_state.critic_clean_reviews[0]["summary"]
        )

    def test_invalid_severity_defaults_to_medium(self):
        """Unknown severity value falls back to MEDIUM."""
        agent = _make_agent()
        agent.research_state = _make_state()
        task = MagicMock()
        resp = _make_response(
            '```json\n{"summary": "s", "details": "d", "critiques": '
            '[{"severity": "CRITICAL", "target_id": "WH-001", "argument": "bad"}]}\n```'
        )
        agent.process_response(resp, task, iteration=1)
        crit_id = list(agent.research_state.critiques.keys())[0]
        assert agent.research_state.critiques[crit_id].severity == Severity.MEDIUM

    def test_strategy_target(self):
        """STRATEGY target does not attempt hypothesis linking."""
        agent = _make_agent()
        agent.research_state = _make_state()
        task = MagicMock()
        resp = _make_response(
            '```json\n{"summary": "s", "details": "d", "critiques": '
            '[{"severity": "MEDIUM", "target_id": "STRATEGY", "argument": "strategic flaw"}]}\n```'
        )
        agent.process_response(resp, task, iteration=1)
        assert not agent._no_critiques_filed
        crit_id = list(agent.research_state.critiques.keys())[0]
        assert agent.research_state.critiques[crit_id].targets == ["STRATEGY"]

    def test_no_research_state(self):
        """No crash when research_state is None."""
        agent = _make_agent()
        agent.research_state = None
        task = MagicMock()
        resp = _make_response(
            '```json\n{"summary": "s", "details": "d", "critiques": '
            '[{"severity": "HIGH", "target_id": "WH-001", "argument": "issue"}]}\n```'
        )
        # Should not raise
        agent.process_response(resp, task, iteration=1)
