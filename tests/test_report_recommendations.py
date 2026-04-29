"""Tests for all report recommendations (P0-A, P0-B, P1-A, P1-B, P2-A, P2-B, P2-C, P2-D, P3).

Covers: zero-text watchdog, checkpoint message, token alert, stale unverified labels,
WH→ER header promotion, computation counter fix, critique resolution regex, and
redundant critic skip.
"""

import re
from unittest.mock import MagicMock, patch

from open_dirac.config import Config, DEFAULTS
from open_dirac.llm import run_agent_loop
from open_dirac.providers.base import ProviderResponse
from open_dirac.research_state import (
    Hypothesis,
    HypothesisStatus,
    ResearchState,
    ReviewResult,
)
from open_dirac.task import Task, TaskType
from open_dirac.agents.computer.tools import ToolExecutor
from open_dirac.validation import (
    check_er_demotion_safety,
    check_stale_unverified_labels,
)


def _safe_path_mock() -> MagicMock:
    """MagicMock safe for ``ws.root.__truediv__`` — see test_engine.py."""
    m = MagicMock()
    m.return_value.exists.return_value = False
    return m


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _mock_provider_response(
    text="", stop_reason="end_turn", input_tokens=100, output_tokens=50, tool_calls=None
):
    """Create a mock ProviderResponse."""
    return ProviderResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason=stop_reason,
        tool_calls=tool_calls,
        raw_content=None,
    )


def _make_config(**overrides) -> Config:
    defaults = dict(
        api_key="test-key",
        logs_dir="",
        provider="anthropic",
        progress_check_interval=999,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_executor():
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp())
    return ToolExecutor(workspace_root=root, timeout=60)


def _mock_provider():
    """Create a mock provider with sensible defaults for format methods."""
    provider = MagicMock()
    provider.format_assistant_message.return_value = {
        "role": "assistant",
        "content": "mock",
    }
    provider.build_tool_result_messages.return_value = [{"role": "user", "content": []}]
    provider.prepare_messages.side_effect = lambda msgs: msgs
    return provider


# ---------------------------------------------------------------------------
# P0-A: Progress Check (replaces zero-text watchdog)
# ---------------------------------------------------------------------------


class TestProgressCheck:
    """P0-A: Progress check injected after N consecutive execute_python rounds."""

    @patch("open_dirac.llm._get_provider")
    def test_progress_check_injected_after_n_rounds(self, mock_get_provider):
        """Progress check message injected after progress_check_interval consecutive exec_python."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        tc = [{"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}]
        submit_tc = [
            {
                "id": "t2",
                "name": "submit_result",
                "input": {
                    "target_id": "RQ-001",
                    "description": "done",
                    "method": "numerical",
                    "result": "ok",
                    "confidence": "exact",
                    "notes": "done",
                },
            }
        ]
        tool_response = _mock_provider_response("", "tool_use", 100, 50, tool_calls=tc)
        text_response = _mock_provider_response("Done.", "end_turn", 100, 50)
        submit_response = _mock_provider_response(
            "", "tool_use", 100, 50, tool_calls=submit_tc
        )

        # 3 tool rounds (triggers progress check), then text end_turn → recovery → submit_result
        provider.call.side_effect = [
            tool_response,
            tool_response,
            tool_response,
            text_response,
            submit_response,
        ]

        config = _make_config(progress_check_interval=3, max_tool_rounds=10)
        executor = _make_executor()
        run_agent_loop(
            system="sys",
            user_content="question",
            config=config,
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=10,
        )

        # Round 4 should see the progress check message
        calls = provider.call.call_args_list
        round4_messages = calls[3].kwargs["messages"]
        progress_found = any(
            isinstance(msg.get("content"), str) and "PROGRESS CHECK" in msg["content"]
            for msg in round4_messages
            if isinstance(msg, dict) and msg.get("role") == "user"
        )
        assert progress_found, (
            "Progress check should be injected after 3 exec_python rounds"
        )

    @patch("open_dirac.llm._get_provider")
    def test_progress_check_resets_after_report_progress(self, mock_get_provider):
        """Calling report_progress resets counter → no injection next round."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        exec_tc = [
            {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
        ]
        progress_tc = [
            {
                "id": "t2",
                "name": "report_progress",
                "input": {
                    "findings_so_far": "ok",
                    "remaining_questions": "",
                    "ready_to_conclude": False,
                },
            }
        ]

        submit_tc = [
            {
                "id": "t3",
                "name": "submit_result",
                "input": {
                    "target_id": "RQ-001",
                    "description": "done",
                    "method": "numerical",
                    "result": "ok",
                    "confidence": "exact",
                    "notes": "done",
                },
            }
        ]

        exec_resp = _mock_provider_response("", "tool_use", 100, 50, tool_calls=exec_tc)
        progress_resp = _mock_provider_response(
            "", "tool_use", 100, 50, tool_calls=progress_tc
        )
        final_resp = _mock_provider_response("Done.", "end_turn", 100, 50)
        submit_resp = _mock_provider_response(
            "", "tool_use", 100, 50, tool_calls=submit_tc
        )

        # 2 exec → progress (resets) → 1 exec → text end_turn → recovery → submit_result
        provider.call.side_effect = [
            exec_resp,
            exec_resp,
            progress_resp,
            exec_resp,
            final_resp,
            submit_resp,
        ]

        config = _make_config(progress_check_interval=2, max_tool_rounds=10)
        executor = _make_executor()
        run_agent_loop(
            system="sys",
            user_content="question",
            config=config,
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=10,
        )

        # Progress check should fire once (after round 2), not after round 4.
        # Count PROGRESS CHECK messages in the LAST call's messages (they accumulate).
        last_messages = provider.call.call_args_list[-1].kwargs["messages"]
        progress_count = sum(
            1
            for msg in last_messages
            if isinstance(msg, dict)
            and isinstance(msg.get("content"), str)
            and "PROGRESS CHECK" in msg["content"]
        )
        assert progress_count == 1

    @patch("open_dirac.llm._get_provider")
    def test_no_progress_check_before_interval(self, mock_get_provider):
        """No progress check if fewer than N exec_python rounds."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        tc = [{"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}]
        submit_tc = [
            {
                "id": "t2",
                "name": "submit_result",
                "input": {
                    "target_id": "RQ-001",
                    "description": "done",
                    "method": "numerical",
                    "result": "ok",
                    "confidence": "exact",
                    "notes": "done",
                },
            }
        ]
        exec_resp = _mock_provider_response("", "tool_use", 100, 50, tool_calls=tc)
        final_resp = _mock_provider_response("Done.", "end_turn", 100, 50)
        submit_resp = _mock_provider_response(
            "", "tool_use", 100, 50, tool_calls=submit_tc
        )

        # Only 2 exec rounds (interval=3), no progress check; text end_turn → recovery → submit
        provider.call.side_effect = [exec_resp, exec_resp, final_resp, submit_resp]

        config = _make_config(progress_check_interval=3, max_tool_rounds=10)
        executor = _make_executor()
        run_agent_loop(
            system="sys",
            user_content="question",
            config=config,
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=10,
        )

        for call in provider.call.call_args_list:
            msgs = call.kwargs.get("messages", [])
            for msg in msgs:
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    assert "PROGRESS CHECK" not in msg["content"]


# ---------------------------------------------------------------------------
# Final Warning Near End of Loop
# ---------------------------------------------------------------------------


class TestFinalWarning:
    """Final warning injected 2 rounds before max_rounds."""

    @patch("open_dirac.llm._get_provider")
    def test_final_warning_injected_near_end(self, mock_get_provider):
        """FINAL WARNING appears in messages at round max_rounds-2."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        tc = [{"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}]
        tool_response = _mock_provider_response(
            "Working on computation...",
            "tool_use",
            100,
            50,
            tool_calls=tc,
        )
        text_response = _mock_provider_response("Done.", "end_turn", 100, 50)

        # 9 tool rounds then end_turn on round 10
        provider.call.side_effect = [tool_response] * 9 + [text_response]

        config = _make_config(max_tool_rounds=10)
        executor = _make_executor()
        run_agent_loop(
            system="sys",
            user_content="question",
            config=config,
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=10,
        )

        # Final warning fires after round 8 (max_rounds - 2 = 8), so round 9's
        # call should see it in messages
        calls = provider.call.call_args_list
        round9_messages = calls[8].kwargs["messages"]
        warning_found = any(
            isinstance(msg.get("content"), str)
            and "You have 2 rounds left" in msg["content"]
            for msg in round9_messages
            if isinstance(msg, dict) and msg.get("role") == "user"
        )
        assert warning_found, "WARNING should be injected after round 8 (max_rounds-2)"

    @patch("open_dirac.llm._get_provider")
    def test_no_final_warning_for_short_loops(self, mock_get_provider):
        """No FINAL WARNING when max_rounds < 5."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        tc = [{"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}]
        tool_response = _mock_provider_response("", "tool_use", 100, 50, tool_calls=tc)
        text_response = _mock_provider_response("Done.", "end_turn", 100, 50)

        # 3 tool rounds then end_turn
        provider.call.side_effect = [tool_response] * 3 + [text_response]

        config = _make_config(max_tool_rounds=4)
        executor = _make_executor()
        run_agent_loop(
            system="sys",
            user_content="question",
            config=config,
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=4,
        )

        # No warning should appear in any call
        for call in provider.call.call_args_list:
            msgs = call.kwargs.get("messages", [])
            for msg in msgs:
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    assert "You have 2 rounds left" not in msg["content"]


# ---------------------------------------------------------------------------
# P1-B: Per-Computation Token Budget Alert
# ---------------------------------------------------------------------------


class TestTokenAlert:
    """P1-B: token_alert_fired set when input tokens exceed threshold."""

    @patch("open_dirac.llm._get_provider")
    def test_alert_fired_above_threshold(self, mock_get_provider):
        """token_alert_fired=True when total_input > computation_token_alert."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Single round with high token count
        text_response = _mock_provider_response(
            "Done.",
            "end_turn",
            input_tokens=200_000,
            output_tokens=100,
        )
        provider.call.return_value = text_response

        config = _make_config(computation_token_alert=150_000)
        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=config,
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=5,
        )

        assert result.token_alert_fired is True

    @patch("open_dirac.llm._get_provider")
    def test_alert_not_fired_below_threshold(self, mock_get_provider):
        """token_alert_fired=False when total_input <= computation_token_alert."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        text_response = _mock_provider_response(
            "Done.",
            "end_turn",
            input_tokens=50_000,
            output_tokens=100,
        )
        provider.call.return_value = text_response

        config = _make_config(computation_token_alert=150_000)
        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=config,
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=5,
        )

        assert result.token_alert_fired is False

    @patch("open_dirac.llm._get_provider")
    def test_alert_accumulates_across_rounds(self, mock_get_provider):
        """Alert fires when cumulative input exceeds threshold."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        tc = [{"id": "t1", "name": "execute_python", "input": {"code": "x=1"}}]
        submit_tc = [
            {
                "id": "t2",
                "name": "submit_result",
                "input": {
                    "target_id": "RQ-001",
                    "description": "done",
                    "method": "numerical",
                    "result": "ok",
                    "confidence": "exact",
                    "notes": "done",
                },
            }
        ]
        r1 = _mock_provider_response(
            "", "tool_use", input_tokens=80_000, output_tokens=50, tool_calls=tc
        )
        r2 = _mock_provider_response(
            "", "tool_use", input_tokens=80_000, output_tokens=50, tool_calls=tc
        )
        r3 = _mock_provider_response(
            "Done.", "end_turn", input_tokens=80_000, output_tokens=50
        )
        r4 = _mock_provider_response(
            "", "tool_use", input_tokens=10_000, output_tokens=50, tool_calls=submit_tc
        )
        provider.call.side_effect = [r1, r2, r3, r4]

        config = _make_config(computation_token_alert=150_000)
        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=config,
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=10,
        )

        # 80K + 80K = 160K > 150K → should fire after round 2
        assert result.token_alert_fired is True
        assert result.total_input_tokens == 250_000  # 80K + 80K + 80K + 10K


# ---------------------------------------------------------------------------
# P2-A: Fix [unverified] Label Persistence
# ---------------------------------------------------------------------------


class TestStaleUnverifiedLabels:
    """P2-A: [unverified] labels promoted to VERIFIED when computation backs them.

    check_stale_unverified_labels operates on ResearchState: it scans
    hypothesis.derivation for [unverified] text and replaces with VERIFIED
    when backed by a VERIFIED computation.  The function always returns []
    (mutations are in-place on state), so tests verify derivation changes.
    """

    def test_unverified_promoted_when_backed(self):
        """[unverified] → VERIFIED when a VERIFIED computation targets the hypothesis."""
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            derivation="ER-001 is [unverified] pending computation.",
        )
        state.hypotheses["ER-001"].review = ReviewResult(
            verdict="VERIFIED",
            iteration=1,
        )
        check_stale_unverified_labels(state)
        assert "VERIFIED" in state.hypotheses["ER-001"].derivation
        assert "[unverified]" not in state.hypotheses["ER-001"].derivation

    def test_unverified_kept_when_no_backing(self):
        """[unverified] stays when no VERIFIED computation exists."""
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            derivation="WH-001 is [unverified] pending.",
        )
        check_stale_unverified_labels(state)
        assert "[unverified]" in state.hypotheses["WH-001"].derivation

    def test_no_unverified_no_action(self):
        """No mutations when no [unverified] labels exist."""
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            derivation="ER-001 result is established.",
        )
        result = check_stale_unverified_labels(state)
        assert result == []
        assert state.hypotheses["ER-001"].derivation == "ER-001 result is established."

    def test_mixed_ids_only_backed_promoted(self):
        """Only hypotheses with VERIFIED backing are promoted."""
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            derivation="ER-001 is [unverified].",
        )
        state.hypotheses["ER-002"] = Hypothesis(
            id="ER-002",
            derivation="ER-002 is [unverified].",
        )
        state.hypotheses["ER-001"].review = ReviewResult(
            verdict="VERIFIED",
            iteration=1,
        )
        check_stale_unverified_labels(state)
        assert "VERIFIED" in state.hypotheses["ER-001"].derivation
        assert "[unverified]" not in state.hypotheses["ER-001"].derivation
        assert "[unverified]" in state.hypotheses["ER-002"].derivation

    def test_wh_renamed_to_er_in_promoted(self):
        """[unverified] promoted and WH-NNN renamed to ER-NNN when WH was promoted to ER."""
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
            derivation="WH-001 was [unverified] but now promoted.",
        )
        state.hypotheses["ER-001"].review = ReviewResult(
            verdict="VERIFIED",
            iteration=1,
        )
        check_stale_unverified_labels(state)
        assert "ER-001" in state.hypotheses["ER-001"].derivation
        assert "WH-001" not in state.hypotheses["ER-001"].derivation
        assert "VERIFIED" in state.hypotheses["ER-001"].derivation
        assert "[unverified]" not in state.hypotheses["ER-001"].derivation

    def test_no_promotion_when_wh_not_promoted_to_er(self):
        """[unverified] stays when WH-002 is verified but ER-002 doesn't exist in state."""
        state = ResearchState()
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            derivation="ER-002 — [unverified] result.",
        )
        state.hypotheses["WH-002"].review = ReviewResult(
            verdict="VERIFIED",
            iteration=1,
        )
        check_stale_unverified_labels(state)
        # The derivation line references ER-002 (not WH-002), and ER-002 doesn't
        # exist in state.  verified_ids = {"WH-002"} (ER-002 not added because
        # it doesn't exist in hypotheses).  The line's id is "ER-002" which is
        # NOT in verified_ids, so [unverified] stays.
        assert "[unverified]" in state.hypotheses["WH-002"].derivation


# ---------------------------------------------------------------------------
# P2-B: Fix WH-to-ER Header Renaming on Promotion
# ---------------------------------------------------------------------------


class TestErDemotionNoAutoPromote:
    """P2-B: check_er_demotion_safety demotes ER→WH on REFUTED computations,
    but does NOT auto-promote WHs.  Operates on ResearchState directly."""

    def test_demotion_fires_when_refuted(self):
        """ER-001 demoted to WH-001 when verification verdict is REFUTED."""
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
            review=ReviewResult(verdict="REFUTED", iteration=1),
        )
        violations = check_er_demotion_safety(state)
        assert len(violations) == 1
        assert "WH-001" in state.hypotheses
        assert "ER-001" not in state.hypotheses

    def test_no_demotion_when_verified(self):
        """ER-001 stays ER when verification verdict is VERIFIED."""
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
            review=ReviewResult(verdict="VERIFIED", iteration=2),
        )
        violations = check_er_demotion_safety(state)
        assert len(violations) == 0
        assert "ER-001" in state.hypotheses

    def test_no_promotion_without_verified(self):
        """WH-001 stays WH — no auto-promotion even with VERIFIED verification."""
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict="VERIFIED", iteration=1),
        )
        violations = check_er_demotion_safety(state)
        assert len(violations) == 0
        # WH-001 stays as WH — no auto-promotion
        assert "WH-001" in state.hypotheses
        assert "ER-001" not in state.hypotheses

    def test_no_demotion_when_no_verification(self):
        """ER-001 stays ER when no verification result exists."""
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            status=HypothesisStatus.ESTABLISHED,
        )
        violations = check_er_demotion_safety(state)
        assert len(violations) == 0
        assert "ER-001" in state.hypotheses


# ---------------------------------------------------------------------------
# P2-D: Fix Critique Resolution Text Truncation
# ---------------------------------------------------------------------------


class TestCritiqueResolutionRegex:
    """P2-D: Multi-line resolution notes captured, not truncated at first period."""

    def test_multiline_resolution_captured(self):
        """Resolution text spanning multiple lines is captured."""
        response_text = (
            "CRIT-001: Corrected the sign error in Eq. 3.\n"
            "The minus sign was missing from the exponent,\n"
            "which caused the divergence issue.\n\n"
            "CRIT-002: Added missing normalization factor."
        )

        # Call the extraction logic directly
        resolution_notes = {}
        for crit_id in ["CRIT-001", "CRIT-002"]:
            note_match = re.search(
                rf"{re.escape(crit_id)}[\s:—\-]+(.+?)(?=\n\n|\nCRIT(?:IQUE)?-\d|$)",
                response_text,
                re.DOTALL,
            )
            if note_match:
                note = note_match.group(1).strip()
                note = " ".join(note.split())
                if len(note) > 300:
                    cut = note[:300].rfind(".")
                    note = note[: cut + 1] if cut > 50 else note[:300] + "..."
                resolution_notes[crit_id] = note

        # CRIT-001 should capture full multi-line text
        assert "CRIT-001" in resolution_notes
        assert "sign error" in resolution_notes["CRIT-001"]
        assert "divergence issue" in resolution_notes["CRIT-001"]

        # CRIT-002 should also be captured
        assert "CRIT-002" in resolution_notes
        assert "normalization" in resolution_notes["CRIT-002"]

    def test_old_regex_would_truncate(self):
        """Verify the old regex would have truncated at first period."""
        text = "CRIT-001: Fixed the error in Eq. 3 which caused divergence."

        # Old regex
        old_match = re.search(
            r"CRIT-001[\s:—\-]+([^.\n]{10,120}[.])",
            text,
        )
        # New regex
        new_match = re.search(
            r"CRIT-001[\s:—\-]+(.+?)(?=\n\n|\nCRIT(?:IQUE)?-\d|$)",
            text,
            re.DOTALL,
        )

        old_text = old_match.group(1).strip() if old_match else ""
        new_text = new_match.group(1).strip() if new_match else ""

        # Old truncates at "Eq." — new captures full sentence
        assert len(new_text) >= len(old_text)
        assert "divergence" in new_text

    def test_long_note_capped_at_300(self):
        """Very long notes are capped at ~300 chars at sentence boundary."""
        long_text = "CRIT-001: " + "This is a sentence. " * 30  # ~600 chars

        note_match = re.search(
            r"CRIT-001[\s:—\-]+(.+?)(?=\n\n|\nCRIT(?:IQUE)?-\d|$)",
            long_text,
            re.DOTALL,
        )
        note = note_match.group(1).strip()
        note = " ".join(note.split())
        if len(note) > 300:
            cut = note[:300].rfind(".")
            note = note[: cut + 1] if cut > 50 else note[:300] + "..."

        assert len(note) <= 303  # 300 + "..."


# ---------------------------------------------------------------------------
# P3: Skip Redundant Critic Passes
# ---------------------------------------------------------------------------


class TestShouldTriggerCritic:
    """P3: _should_trigger_critic fires after VERIFIED review when interval exceeded."""

    def _make_engine(
        self,
        last_critic_iteration=0,
        last_verified_review_iteration=0,
        current_iteration=5,
        critic_every_n=4,
    ):
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"
            ws.read_file = MagicMock(return_value="")

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config(critic_every_n=critic_every_n)
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.metrics.last_critic_iteration = last_critic_iteration
            engine.iteration = current_iteration
            from open_dirac.engine import LoopState

            engine._state = LoopState(
                last_verified_review_iteration=last_verified_review_iteration,
            )

        return engine

    def test_trigger_after_verified_review(self):
        """Triggers on every VERIFIED review, regardless of interval."""
        engine = self._make_engine(
            last_critic_iteration=0,
            last_verified_review_iteration=5,
            current_iteration=5,
            critic_every_n=4,
        )
        assert engine._should_trigger_critic() is True

    def test_trigger_even_when_critic_recent(self):
        """Triggers even when critic ran recently — no delay constraint."""
        engine = self._make_engine(
            last_critic_iteration=4,
            last_verified_review_iteration=5,
            current_iteration=5,
            critic_every_n=4,
        )
        assert engine._should_trigger_critic() is True

    def test_no_trigger_without_verified_review(self):
        """Does not trigger when latest iteration is not a VERIFIED review."""
        engine = self._make_engine(
            last_critic_iteration=0,
            last_verified_review_iteration=3,
            current_iteration=5,
            critic_every_n=4,
        )
        assert engine._should_trigger_critic() is False

    def test_last_content_tracked_in_dispatch(self):
        """_dispatch updates _last_content_iteration for research/compute tasks."""
        with patch("open_dirac.engine.WorkspaceManager") as MockWS:
            ws = MockWS.return_value
            ws.init = MagicMock()
            ws.root = MagicMock()
            ws.root.__truediv__ = _safe_path_mock()
            ws.logs_dir = "/tmp/logs"
            ws.read_file = MagicMock(return_value="")

            from open_dirac.engine import OpenDirac

            engine = OpenDirac.__new__(OpenDirac)
            engine.config = Config()
            engine.workspace = ws
            engine.metrics = MagicMock()
            engine.iteration = 7
            from open_dirac.engine import LoopState
            from open_dirac.research_state import ResearchState

            engine._state = LoopState()
            engine.research_state = ResearchState()

            engine.researcher = MagicMock()
            engine.computer = MagicMock()
            engine.critic = MagicMock()

        task = Task(task_id="T", task_type=TaskType.RESEARCH, assigned_to="researcher")
        engine._dispatch(task)
        assert engine._state.last_content_iteration == 7


# ---------------------------------------------------------------------------
# Config: new fields present
# ---------------------------------------------------------------------------


class TestNewConfigFields:
    """Verify config fields are properly loaded from defaults."""

    def test_progress_check_interval_default(self):
        assert DEFAULTS["progress_check_interval"] == 4
        assert Config().progress_check_interval == 4

    def test_computation_token_alert_default(self):
        assert DEFAULTS["computation_token_alert"] == 150_000
        assert Config().computation_token_alert == 150_000

    def test_new_fields_in_yaml_config_fields(self):
        from open_dirac.config import _YAML_CONFIG_FIELDS

        assert "progress_check_interval" in _YAML_CONFIG_FIELDS
        assert "computation_token_alert" in _YAML_CONFIG_FIELDS
