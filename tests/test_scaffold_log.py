"""Tests for EVENT_LOG.jsonl instrumentation."""

import json


from open_dirac.config import Config
from open_dirac.workspace import log_llm_call, log_scaffold_event


class TestLogScaffoldEvent:
    """Core function tests for log_scaffold_event."""

    def test_creates_valid_jsonl(self, tmp_path):
        log_scaffold_event(
            tmp_path,
            iteration=2,
            category="loop_control",
            event="p1_budget_override",
            detail="compute -> synthesize",
        )
        logfile = tmp_path / "EVENT_LOG.jsonl"
        assert logfile.exists()
        entry = json.loads(logfile.read_text().strip())
        assert entry["kind"] == "scaffold"
        assert entry["iter"] == 2
        assert entry["category"] == "loop_control"
        assert entry["event"] == "p1_budget_override"
        assert entry["detail"] == "compute -> synthesize"
        assert "ts" in entry

    def test_appends_multiple(self, tmp_path):
        log_scaffold_event(
            tmp_path, 1, "state_invariants", "phantom_references", "first"
        )
        log_scaffold_event(tmp_path, 2, "loop_control", "p3_forced_critic", "second")
        lines = (tmp_path / "EVENT_LOG.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "phantom_references"
        assert json.loads(lines[1])["event"] == "p3_forced_critic"

    def test_no_crash_on_bad_path(self):
        # Writing to a nonexistent directory should not raise
        log_scaffold_event(
            "/nonexistent/dir/xyz", 1, "call_reliability", "api_retry", "test"
        )

    def test_empty_detail(self, tmp_path):
        log_scaffold_event(tmp_path, 1, "output_normalization", "preamble_stripped")
        entry = json.loads((tmp_path / "EVENT_LOG.jsonl").read_text().strip())
        assert entry["detail"] == ""


class TestValidatePostIntegrationLogs:
    """Integration test: validate_post_integration writes state_invariants events."""

    def test_violations_are_logged(self, tmp_path):
        """Set up state with phantom VERIFIED label, verify EVENT_LOG.jsonl gets state_invariants event."""
        from open_dirac.workspace import WorkspaceManager
        from open_dirac.validation import validate_post_integration
        from open_dirac.research_state import ResearchState, Hypothesis

        config = Config(workspace_dir=str(tmp_path / "ws"))
        ws = WorkspaceManager(config)
        ws.init("Test problem.")

        # Create state with unsubstantiated VERIFIED in derivation
        research_state = ResearchState()
        research_state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            derivation="WH-001 is VERIFIED by computation.",
        )

        violations = validate_post_integration(
            research_state, iteration=3, workspace=ws
        )
        assert len(violations) > 0

        logfile = ws.root / "EVENT_LOG.jsonl"
        assert logfile.exists()
        entries = [json.loads(line) for line in logfile.read_text().strip().split("\n")]
        state_inv = [e for e in entries if e["category"] == "state_invariants"]
        assert len(state_inv) > 0
        assert any(e["event"] == "phantom_labels" for e in state_inv)
        assert all(e["iter"] == 3 for e in state_inv)


class TestLogLlmCall:
    """Tests for the log_llm_call function."""

    def test_creates_valid_jsonl(self, tmp_path):
        log_llm_call(
            tmp_path,
            agent="orchestrator",
            iteration=1,
            model="claude-sonnet-4-6",
            input_tokens=1908,
            output_tokens=802,
            stop_reason="end_turn",
            duration_s=8.53,
            system_prompt_chars=5000,
            user_content_chars=3000,
            response_chars=1500,
        )
        logfile = tmp_path / "EVENT_LOG.jsonl"
        assert logfile.exists()
        entry = json.loads(logfile.read_text().strip())
        assert entry["kind"] == "llm_call"
        assert entry["agent"] == "orchestrator"
        assert entry["iter"] == 1
        assert entry["model"] == "claude-sonnet-4-6"
        assert entry["input_tokens"] == 1908
        assert entry["output_tokens"] == 802
        assert entry["stop_reason"] == "end_turn"
        assert entry["duration_s"] == 8.53
        assert entry["reasoning_tokens"] == 0
        assert entry["round"] == 0
        assert "ts" in entry

    def test_interleaves_with_scaffold_events(self, tmp_path):
        log_scaffold_event(tmp_path, 1, "loop_control", "p1_budget_override", "first")
        log_llm_call(
            tmp_path,
            agent="researcher",
            iteration=1,
            model="m",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration_s=1.0,
            system_prompt_chars=100,
            user_content_chars=200,
            response_chars=50,
        )
        log_scaffold_event(tmp_path, 2, "loop_control", "dispatch_failure", "second")
        lines = (tmp_path / "EVENT_LOG.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["kind"] == "scaffold"
        assert json.loads(lines[1])["kind"] == "llm_call"
        assert json.loads(lines[2])["kind"] == "scaffold"

    def test_no_crash_on_bad_path(self):
        log_llm_call(
            "/nonexistent/dir/xyz",
            agent="test",
            iteration=1,
            model="m",
            input_tokens=0,
            output_tokens=0,
            stop_reason="end_turn",
            duration_s=0.0,
            system_prompt_chars=0,
            user_content_chars=0,
            response_chars=0,
        )

    def test_reasoning_tokens_and_round(self, tmp_path):
        log_llm_call(
            tmp_path,
            agent="computationalist",
            iteration=3,
            model="m",
            input_tokens=500,
            output_tokens=200,
            stop_reason="tool_use",
            duration_s=2.5,
            system_prompt_chars=1000,
            user_content_chars=2000,
            response_chars=500,
            reasoning_tokens=100,
            answer_tokens=100,
            round=2,
        )
        entry = json.loads((tmp_path / "EVENT_LOG.jsonl").read_text().strip())
        assert entry["reasoning_tokens"] == 100
        assert entry["answer_tokens"] == 100
        assert entry["round"] == 2
