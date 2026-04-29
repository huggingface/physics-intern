"""Tests for transient-error retry logic in llm.py."""

from unittest.mock import MagicMock, patch

import pytest

from open_dirac.config import Config
from open_dirac.llm import _call_provider_with_retry
from open_dirac.providers.retry import (
    is_transient as _is_transient,
    is_tool_call_failure as _is_tool_call_failure,
    is_provider_side_400 as _is_provider_side_400,
    extract_status_code as _extract_status_code,
)
from open_dirac.providers.base import ProviderResponse


# ---------------------------------------------------------------------------
# _is_transient classifier
# ---------------------------------------------------------------------------


class FakeHTTPError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class FakeStatusError(Exception):
    """Uses .status instead of .status_code (some SDKs)."""

    def __init__(self, status):
        self.status = status
        super().__init__(f"status {status}")


class FakeGoogleServerError(Exception):
    """Mimics google.genai.errors.ServerError: .status is a string, .code holds the int."""

    def __init__(self, code, status_str):
        self.code = code
        self.status = status_str
        super().__init__(f"{code} {status_str}")


FakeGoogleServerError.__name__ = "ServerError"


class FakeConnectionError(ConnectionError):
    pass


class FakeTimeoutError(TimeoutError):
    pass


class FakeReadTimeout(Exception):
    """Mimics requests.exceptions.ReadTimeout naming."""

    pass


FakeReadTimeout.__name__ = "ReadTimeout"


class FakeResponseStatusError(Exception):
    """Mimics httpx/huggingface_hub pattern: status on exc.response.status_code."""

    def __init__(self, status_code):
        self.response = MagicMock(status_code=status_code)
        super().__init__(f"Server error '{status_code}'")


class FakeAuthError(Exception):
    def __init__(self):
        self.status_code = 401
        super().__init__("Unauthorized")


@pytest.mark.parametrize(
    "exc",
    [
        FakeHTTPError(429),
        FakeHTTPError(500),
        FakeHTTPError(502),
        FakeHTTPError(503),
        FakeHTTPError(504),
        FakeStatusError(429),
        FakeStatusError(500),
        FakeStatusError(504),
        FakeResponseStatusError(500),
        FakeResponseStatusError(502),
        FakeResponseStatusError(503),
        FakeResponseStatusError(504),
        FakeGoogleServerError(502, "Bad Gateway"),
        FakeGoogleServerError(503, "Service Unavailable"),
        FakeConnectionError("connection reset"),
        FakeTimeoutError("timed out"),
        FakeReadTimeout(),
        ConnectionError("reset by peer"),
        TimeoutError("deadline exceeded"),
        type("RemoteProtocolError", (Exception,), {})(
            "peer closed connection without sending complete message body"
        ),
    ],
)
def test_is_transient_true(exc):
    assert _is_transient(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        FakeHTTPError(400),
        FakeHTTPError(401),
        FakeHTTPError(403),
        FakeHTTPError(404),
        FakeHTTPError(422),
        FakeAuthError(),
        FakeResponseStatusError(400),
        FakeResponseStatusError(422),
        ValueError("bad input"),
        RuntimeError("something else"),
        TypeError("wrong type"),
    ],
)
def test_is_transient_false(exc):
    assert _is_transient(exc) is False


# ---------------------------------------------------------------------------
# _extract_status_code
# ---------------------------------------------------------------------------


def test_extract_status_code_from_status_code_attr():
    assert _extract_status_code(FakeHTTPError(502)) == 502


def test_extract_status_code_from_status_attr_int():
    assert _extract_status_code(FakeStatusError(429)) == 429


def test_extract_status_code_from_response_attr():
    assert _extract_status_code(FakeResponseStatusError(503)) == 503


def test_extract_status_code_skips_string_status():
    """Google ServerError: .status='Bad Gateway', .code=502 → returns 502."""
    assert _extract_status_code(FakeGoogleServerError(502, "Bad Gateway")) == 502


def test_extract_status_code_none_when_no_code():
    assert _extract_status_code(ValueError("no status")) is None


# ---------------------------------------------------------------------------
# _is_tool_call_failure
# ---------------------------------------------------------------------------


class FakeToolCallFailure(Exception):
    """Mimics HuggingFace BadRequestError with tool_use_failed code."""

    def __init__(self):
        self.status_code = 400
        super().__init__(
            "BadRequestError: tool_use_failed - "
            "Failed to parse tool call arguments as valid JSON"
        )


def test_is_tool_call_failure_true():
    assert _is_tool_call_failure(FakeToolCallFailure()) is True


def test_is_tool_call_failure_false_on_regular_400():
    assert _is_tool_call_failure(FakeHTTPError(400)) is False


def test_is_transient_true_for_tool_call_failure():
    """_is_transient returns True for tool_use_failed errors (stochastic retry)."""
    assert _is_transient(FakeToolCallFailure()) is True


class FakeOutputParseError(Exception):
    """Mimics HF 'output_parse_failed' when model ignores tool_choice=none."""

    def __init__(self):
        self.status_code = 400
        super().__init__(
            "BadRequestError: output_parse_failed - "
            "Parsing failed. The model generated output that could not be parsed."
        )


class FakeToolChoiceError(Exception):
    """Mimics HF 'Tool choice is none, but model called a tool'."""

    def __init__(self):
        self.status_code = 400
        super().__init__(
            "BadRequestError: tool_use_failed - "
            "Tool choice is none, but model called a tool"
        )


@pytest.mark.parametrize(
    "exc",
    [
        FakeOutputParseError(),
        FakeToolChoiceError(),
    ],
)
def test_is_tool_call_failure_oss_model_patterns(exc):
    """OSS model tool-choice violations are recognized as tool-call failures."""
    assert _is_tool_call_failure(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        FakeOutputParseError(),
        FakeToolChoiceError(),
    ],
)
def test_is_transient_true_for_oss_model_errors(exc):
    """OSS model tool-choice errors are transient (retryable)."""
    assert _is_transient(exc) is True


# ---------------------------------------------------------------------------
# _is_provider_side_400
# ---------------------------------------------------------------------------


class FakePostProcessorError(Exception):
    """Mimics HF 400 with 'gpt oss post processor' message."""

    def __init__(self):
        self.status_code = 400
        super().__init__(
            "BadRequestError: Encountered Exception during gpt oss post processor"
        )


class FakeInternalError400(Exception):
    """Mimics provider 400 with 'internal error' message."""

    def __init__(self):
        self.status_code = 400
        super().__init__("BadRequestError: internal error during processing")


class FakeBackendError400(Exception):
    """Mimics provider 400 with 'backend error' message."""

    def __init__(self):
        self.status_code = 400
        super().__init__("BadRequestError: backend error in model output")


class FakePostProcessorResponseError(Exception):
    """Mimics httpx-style error where status is on .response.status_code."""

    def __init__(self):
        self.response = MagicMock(status_code=400)
        super().__init__("Encountered Exception during gpt oss post processor")


@pytest.mark.parametrize(
    "exc",
    [
        FakePostProcessorError(),
        FakeInternalError400(),
        FakeBackendError400(),
        FakePostProcessorResponseError(),
    ],
)
def test_is_provider_side_400_true(exc):
    assert _is_provider_side_400(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        FakeHTTPError(400),  # plain 400 without matching message
        FakeHTTPError(500),  # 500 is not a 400
        FakeToolCallFailure(),  # tool_use_failed is a different category
        ValueError("post processor"),  # no status code
        RuntimeError("internal error"),  # no status code
    ],
)
def test_is_provider_side_400_false(exc):
    assert _is_provider_side_400(exc) is False


def test_is_transient_true_for_provider_side_400():
    """Provider-side 400s are classified as transient (retryable, but capped)."""
    assert _is_transient(FakePostProcessorError()) is True


# ---------------------------------------------------------------------------
# _call_provider_with_retry
# ---------------------------------------------------------------------------


def _make_provider_response(text="ok"):
    return ProviderResponse(
        text=text,
        input_tokens=10,
        output_tokens=5,
        stop_reason="end_turn",
    )


def _make_config(**overrides):
    defaults = dict(
        api_retry_max=3,
        api_retry_initial_delay=0.01,  # fast for tests
        api_retry_max_delay=0.1,
        progress_check_interval=999,  # disable progress checks in tests
    )
    defaults.update(overrides)
    return Config(**defaults)


def test_retry_succeeds_after_transient_errors():
    """Provider fails twice with 503, then succeeds on third attempt."""
    provider = MagicMock()
    provider.call.side_effect = [
        FakeHTTPError(503),
        FakeHTTPError(503),
        _make_provider_response("success"),
    ]
    config = _make_config()

    with patch("open_dirac.providers.retry.time.sleep") as mock_sleep:
        result = _call_provider_with_retry(
            provider,
            config,
            model="m",
            max_tokens=100,
            system="s",
            messages=[],
        )

    assert result.text == "success"
    assert provider.call.call_count == 3
    assert mock_sleep.call_count == 2


def test_retry_respects_backoff():
    """Verify exponential backoff with delay doubling."""
    provider = MagicMock()
    provider.call.side_effect = [
        FakeHTTPError(429),
        FakeHTTPError(429),
        FakeHTTPError(429),
        _make_provider_response("ok"),
    ]
    config = _make_config(api_retry_initial_delay=1.0, api_retry_max_delay=10.0)

    with patch("open_dirac.providers.retry.time.sleep") as mock_sleep:
        result = _call_provider_with_retry(
            provider,
            config,
            model="m",
            max_tokens=100,
            system="s",
            messages=[],
        )

    assert result.text == "ok"
    # Delays: 1.0, 2.0, 4.0
    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert delays == [1.0, 2.0, 4.0]


def test_retry_caps_at_max_delay():
    """Backoff delay is capped at api_retry_max_delay."""
    provider = MagicMock()
    provider.call.side_effect = [
        FakeHTTPError(504),
        FakeHTTPError(504),
        FakeHTTPError(504),
        _make_provider_response("ok"),
    ]
    config = _make_config(api_retry_initial_delay=5.0, api_retry_max_delay=8.0)

    with patch("open_dirac.providers.retry.time.sleep") as mock_sleep:
        _call_provider_with_retry(
            provider,
            config,
            model="m",
            max_tokens=100,
            system="s",
            messages=[],
        )

    # Delays: min(5, 8)=5, min(10, 8)=8, min(20, 8)=8
    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert delays == [5.0, 8.0, 8.0]


def test_no_retry_on_non_transient_error():
    """Non-transient errors are raised immediately without retry."""
    provider = MagicMock()
    provider.call.side_effect = FakeHTTPError(400)
    config = _make_config()

    with pytest.raises(FakeHTTPError):
        _call_provider_with_retry(
            provider,
            config,
            model="m",
            max_tokens=100,
            system="s",
            messages=[],
        )

    assert provider.call.call_count == 1


def test_exhausted_retries_raises():
    """After all retries are exhausted, the last transient error is raised."""
    provider = MagicMock()
    provider.call.side_effect = FakeHTTPError(503)
    config = _make_config(api_retry_max=2)

    with patch("open_dirac.providers.retry.time.sleep"):
        with pytest.raises(FakeHTTPError):
            _call_provider_with_retry(
                provider,
                config,
                model="m",
                max_tokens=100,
                system="s",
                messages=[],
            )

    # 1 initial + 2 retries = 3
    assert provider.call.call_count == 3


def test_no_retry_when_max_is_zero():
    """api_retry_max=0 means no retries at all."""
    provider = MagicMock()
    provider.call.side_effect = FakeHTTPError(503)
    config = _make_config(api_retry_max=0)

    with pytest.raises(FakeHTTPError):
        _call_provider_with_retry(
            provider,
            config,
            model="m",
            max_tokens=100,
            system="s",
            messages=[],
        )

    assert provider.call.call_count == 1


def test_immediate_success_no_sleep():
    """When the first call succeeds, no sleep occurs."""
    provider = MagicMock()
    provider.call.return_value = _make_provider_response("instant")
    config = _make_config()

    with patch("open_dirac.providers.retry.time.sleep") as mock_sleep:
        result = _call_provider_with_retry(
            provider,
            config,
            model="m",
            max_tokens=100,
            system="s",
            messages=[],
        )

    assert result.text == "instant"
    assert provider.call.call_count == 1
    mock_sleep.assert_not_called()


def test_retry_with_connection_error():
    """ConnectionError is retried."""
    provider = MagicMock()
    provider.call.side_effect = [
        ConnectionError("reset by peer"),
        _make_provider_response("recovered"),
    ]
    config = _make_config()

    with patch("open_dirac.providers.retry.time.sleep"):
        result = _call_provider_with_retry(
            provider,
            config,
            model="m",
            max_tokens=100,
            system="s",
            messages=[],
        )

    assert result.text == "recovered"
    assert provider.call.call_count == 2


def test_retry_with_timeout_error():
    """TimeoutError is retried."""
    provider = MagicMock()
    provider.call.side_effect = [
        TimeoutError("deadline exceeded"),
        _make_provider_response("ok"),
    ]
    config = _make_config()

    with patch("open_dirac.providers.retry.time.sleep"):
        result = _call_provider_with_retry(
            provider,
            config,
            model="m",
            max_tokens=100,
            system="s",
            messages=[],
        )

    assert result.text == "ok"


def test_provider_side_400_capped_at_2_attempts():
    """Provider-side 400s are retried once, then raised (2 attempts total)."""
    provider = MagicMock()
    provider.call.side_effect = FakePostProcessorError()
    config = _make_config(api_retry_max=10)  # would do 11 attempts normally

    with patch("open_dirac.providers.retry.time.sleep"):
        with pytest.raises(FakePostProcessorError):
            _call_provider_with_retry(
                provider,
                config,
                model="m",
                max_tokens=100,
                system="s",
                messages=[],
            )

    # Only 2 attempts: initial + 1 retry, NOT 11
    assert provider.call.call_count == 2


def test_provider_side_400_succeeds_on_retry():
    """Provider-side 400 on first attempt, success on retry."""
    provider = MagicMock()
    provider.call.side_effect = [
        FakePostProcessorError(),
        _make_provider_response("recovered"),
    ]
    config = _make_config(api_retry_max=10)

    with patch("open_dirac.providers.retry.time.sleep"):
        result = _call_provider_with_retry(
            provider,
            config,
            model="m",
            max_tokens=100,
            system="s",
            messages=[],
        )

    assert result.text == "recovered"
    assert provider.call.call_count == 2


# ---------------------------------------------------------------------------
# Fix 2: Penultimate-round CRITICAL message & forced final prompt
# ---------------------------------------------------------------------------


class TestPenultimateRoundMessage:
    """Test CRITICAL message injection and forced final prompt (Fix 2)."""

    def _make_tool_use_response(self, text="", stop_reason="tool_use"):
        """Create a ProviderResponse that mimics tool_use."""
        return ProviderResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason=stop_reason,
            tool_calls=[
                {"id": "tc_1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
            raw_content=[
                {
                    "type": "tool_use",
                    "id": "tc_1",
                    "name": "execute_python",
                    "input": {"code": "print(1)"},
                }
            ],
        )

    def _make_final_response(self, text="## COMP-001\n**VERDICT:** INCONCLUSIVE"):
        return ProviderResponse(
            text=text,
            input_tokens=100,
            output_tokens=200,
            stop_reason="end_turn",
        )

    def test_critical_message_at_penultimate_round(self):
        """CRITICAL message appears at round max_rounds - 1 when max_rounds >= 4."""
        from open_dirac.llm import run_agent_loop
        from open_dirac.agents.computer.tools import ToolExecutor
        from open_dirac.tool_call import ToolCall

        max_rounds = 5
        provider = MagicMock()
        provider.prepare_messages.side_effect = lambda msgs: msgs
        # Rounds 1..5 return tool_use, then forced final call returns text
        tool_responses = [self._make_tool_use_response() for _ in range(max_rounds)]
        final_response = self._make_final_response()

        provider.call = MagicMock(side_effect=tool_responses + [final_response])
        provider.format_assistant_message = MagicMock(
            return_value={"role": "assistant", "content": "tool"}
        )
        provider.build_tool_result_messages = MagicMock(
            return_value=[{"role": "user", "content": "result"}]
        )

        tool_executor = MagicMock(spec=ToolExecutor)
        tool_executor.execute = MagicMock(
            return_value=ToolCall(
                tool_name="execute_python",
                tool_input={"code": "1"},
                output="1",
                is_error=False,
                duration=0.1,
            )
        )

        config = _make_config()
        config.max_tokens = 4096

        config.logs_dir = ""
        config.computation_token_alert = 999999
        config.progress_check_interval = 999

        with patch("open_dirac.llm._get_provider", return_value=provider):
            result = run_agent_loop(
                system="test",
                user_content="test",
                config=config,
                tool_executor=tool_executor,
                tools=[{"type": "function", "function": {"name": "execute_python"}}],
                max_rounds=max_rounds,
            )

        # Check that CRITICAL message was injected in messages
        all_calls = provider.call.call_args_list
        critical_found = False
        for call in all_calls[:-1]:  # exclude final forced call
            msgs = call.kwargs.get("messages", []) or []
            for msg in msgs:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "CRITICAL" in block.get(
                            "text", ""
                        ):
                            critical_found = True
                elif isinstance(content, str) and "CRITICAL" in content:
                    critical_found = True
        assert critical_found, "CRITICAL message should appear in messages"

    def test_critical_message_not_injected_when_max_rounds_too_small(self):
        """CRITICAL message does NOT appear when max_rounds < 4."""
        from open_dirac.llm import run_agent_loop
        from open_dirac.agents.computer.tools import ToolExecutor
        from open_dirac.tool_call import ToolCall

        max_rounds = 3
        provider = MagicMock()
        provider.prepare_messages.side_effect = lambda msgs: msgs
        tool_responses = [self._make_tool_use_response() for _ in range(max_rounds)]
        final_response = self._make_final_response()

        provider.call = MagicMock(side_effect=tool_responses + [final_response])
        provider.format_assistant_message = MagicMock(
            return_value={"role": "assistant", "content": "tool"}
        )
        provider.build_tool_result_messages = MagicMock(
            return_value=[{"role": "user", "content": "result"}]
        )

        tool_executor = MagicMock(spec=ToolExecutor)
        tool_executor.execute = MagicMock(
            return_value=ToolCall(
                tool_name="execute_python",
                tool_input={"code": "1"},
                output="1",
                is_error=False,
                duration=0.1,
            )
        )

        config = _make_config()
        config.max_tokens = 4096

        config.logs_dir = ""
        config.computation_token_alert = 999999
        config.progress_check_interval = 999

        with patch("open_dirac.llm._get_provider", return_value=provider):
            result = run_agent_loop(
                system="test",
                user_content="test",
                config=config,
                tool_executor=tool_executor,
                tools=[{"type": "function", "function": {"name": "execute_python"}}],
                max_rounds=max_rounds,
            )

        # Check that CRITICAL message was NOT injected
        all_calls = provider.call.call_args_list
        for call in all_calls[:-1]:
            msgs = call.kwargs.get("messages", []) or []
            for msg in msgs:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            assert "CRITICAL" not in block.get("text", ""), (
                                "CRITICAL should not appear when max_rounds < 4"
                            )

    def test_forced_final_call_exception_returns_empty_text(self):
        """When the forced final call raises, result.text is empty (honest failure)."""
        from open_dirac.llm import run_agent_loop
        from open_dirac.agents.computer.tools import ToolExecutor
        from open_dirac.tool_call import ToolCall

        max_rounds = 3
        provider = MagicMock()
        provider.prepare_messages.side_effect = lambda msgs: msgs
        # Rounds 1..3 return tool_use, forced final call raises
        tool_responses = [self._make_tool_use_response() for _ in range(max_rounds)]
        forced_exc = Exception("output_parse_failed - model gibberish")
        forced_exc.status_code = 400

        provider.call = MagicMock(side_effect=tool_responses + [forced_exc])
        provider.format_assistant_message = MagicMock(
            return_value={"role": "assistant", "content": "tool"}
        )
        provider.build_tool_result_messages = MagicMock(
            return_value=[{"role": "user", "content": "result"}]
        )

        tool_executor = MagicMock(spec=ToolExecutor)
        tool_executor.execute = MagicMock(
            return_value=ToolCall(
                tool_name="execute_python",
                tool_input={"code": "print(42)"},
                output="42",
                is_error=False,
                duration=0.1,
            )
        )

        config = _make_config(api_retry_max=0)
        config.max_tokens = 4096

        config.logs_dir = ""
        config.computation_token_alert = 999999
        config.progress_check_interval = 999

        with patch("open_dirac.llm._get_provider", return_value=provider):
            result = run_agent_loop(
                system="test",
                user_content="test",
                config=config,
                tool_executor=tool_executor,
                tools=[{"type": "function", "function": {"name": "execute_python"}}],
                max_rounds=max_rounds,
            )

        # Honest failure: empty text, no synthetic recovery
        assert result.stop_reason == "max_rounds_forced"
        assert result.text == ""
        # Single forced call attempt (no retry)
        assert provider.call.call_count == max_rounds + 1

    def test_progress_check_does_not_break_loop(self):
        """Progress check injection does not break the agent loop."""
        from open_dirac.llm import run_agent_loop
        from open_dirac.agents.computer.tools import ToolExecutor
        from open_dirac.tool_call import ToolCall

        max_rounds = 5
        provider = MagicMock()
        provider.prepare_messages.side_effect = lambda msgs: msgs
        tool_responses = [self._make_tool_use_response() for _ in range(max_rounds)]
        final_response = self._make_final_response()

        provider.call = MagicMock(side_effect=tool_responses + [final_response])
        provider.format_assistant_message = MagicMock(
            return_value={"role": "assistant", "content": "tool"}
        )
        provider.build_tool_result_messages = MagicMock(
            return_value=[{"role": "user", "content": "result"}]
        )

        tool_executor = MagicMock(spec=ToolExecutor)
        tool_executor.execute = MagicMock(
            return_value=ToolCall(
                tool_name="execute_python",
                tool_input={"code": "1"},
                output="1",
                is_error=False,
                duration=0.1,
            )
        )

        config = _make_config(api_retry_max=0)
        config.max_tokens = 4096
        config.logs_dir = ""
        config.computation_token_alert = 999999
        config.progress_check_interval = 2  # fires after 2 consecutive exec_python

        with patch("open_dirac.llm._get_provider", return_value=provider):
            result = run_agent_loop(
                system="test",
                user_content="test",
                config=config,
                tool_executor=tool_executor,
                tools=[{"type": "function", "function": {"name": "execute_python"}}],
                max_rounds=max_rounds,
            )

        # Should complete without crashing
        assert result.text

    def test_tool_call_failure_graceful_degradation(self):
        """run_agent_loop degrades to forced text-only call on tool_use_failed error."""
        from open_dirac.llm import run_agent_loop
        from open_dirac.agents.computer.tools import ToolExecutor

        max_rounds = 5
        provider = MagicMock()
        provider.prepare_messages.side_effect = lambda msgs: msgs

        # Provider raises tool_use_failed on the very first round (after retries)
        tool_fail_exc = Exception(
            "BadRequestError: tool_use_failed - Failed to parse tool call arguments"
        )
        tool_fail_exc.status_code = 400
        final_response = self._make_final_response(
            "## COMP-001: Fallback\n**VERDICT:** INCONCLUSIVE"
        )
        # First call raises, second call (forced text-only) succeeds
        provider.call = MagicMock(side_effect=[tool_fail_exc, final_response])
        provider.format_assistant_message = MagicMock(
            return_value={"role": "assistant", "content": "tool"}
        )
        provider.build_tool_result_messages = MagicMock(
            return_value=[{"role": "user", "content": "result"}]
        )

        tool_executor = MagicMock(spec=ToolExecutor)
        config = _make_config(api_retry_max=0)  # no retries — fail immediately
        config.max_tokens = 4096

        config.logs_dir = ""
        config.computation_token_alert = 999999
        config.progress_check_interval = 999

        with (
            patch("open_dirac.llm._get_provider", return_value=provider),
            patch("open_dirac.providers.retry.time.sleep"),
        ):
            result = run_agent_loop(
                system="test",
                user_content="test",
                config=config,
                tool_executor=tool_executor,
                tools=[{"type": "function", "function": {"name": "execute_python"}}],
                max_rounds=max_rounds,
            )

        assert result.stop_reason == "max_rounds_forced"
        assert "COMP-001" in result.text
        # The forced text-only call should NOT include tools
        final_call = provider.call.call_args_list[-1]
        assert "tools" not in final_call.kwargs

    def test_provider_side_400_graceful_degradation(self):
        """run_agent_loop degrades to forced text-only call on provider-side 400."""
        from open_dirac.llm import run_agent_loop
        from open_dirac.agents.computer.tools import ToolExecutor

        max_rounds = 5
        provider = MagicMock()
        provider.prepare_messages.side_effect = lambda msgs: msgs

        # Provider raises post-processor error on the very first round
        post_proc_exc = FakePostProcessorError()
        final_response = self._make_final_response(
            "## COMP-001: Fallback\n**VERDICT:** INCONCLUSIVE"
        )
        # First call raises, second call (forced text-only) succeeds
        provider.call = MagicMock(side_effect=[post_proc_exc, final_response])
        provider.format_assistant_message = MagicMock(
            return_value={"role": "assistant", "content": "tool"}
        )
        provider.build_tool_result_messages = MagicMock(
            return_value=[{"role": "user", "content": "result"}]
        )

        tool_executor = MagicMock(spec=ToolExecutor)
        config = _make_config(api_retry_max=0)  # no retries — fail immediately
        config.max_tokens = 4096
        config.logs_dir = ""
        config.computation_token_alert = 999999
        config.progress_check_interval = 999

        with (
            patch("open_dirac.llm._get_provider", return_value=provider),
            patch("open_dirac.providers.retry.time.sleep"),
        ):
            result = run_agent_loop(
                system="test",
                user_content="test",
                config=config,
                tool_executor=tool_executor,
                tools=[{"type": "function", "function": {"name": "execute_python"}}],
                max_rounds=max_rounds,
            )

        assert result.stop_reason == "max_rounds_forced"
        assert "COMP-001" in result.text
        # The forced text-only call should NOT include tools
        final_call = provider.call.call_args_list[-1]
        assert "tools" not in final_call.kwargs
        # The forced message should mention provider-side processing error
        final_messages = final_call.kwargs.get("messages", [])
        forced_msgs = [
            m
            for m in final_messages
            if isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and "provider-side processing error" in m["content"]
        ]
        assert len(forced_msgs) == 1

    def test_forced_final_call_uses_user_message_not_system_mutation(self):
        """The forced text-only call uses a user message, not a mutated system prompt."""
        from open_dirac.llm import run_agent_loop
        from open_dirac.agents.computer.tools import ToolExecutor
        from open_dirac.tool_call import ToolCall

        max_rounds = 2
        provider = MagicMock()
        provider.prepare_messages.side_effect = lambda msgs: msgs
        tool_responses = [self._make_tool_use_response() for _ in range(max_rounds)]
        final_response = self._make_final_response()

        provider.call = MagicMock(side_effect=tool_responses + [final_response])
        provider.format_assistant_message = MagicMock(
            return_value={"role": "assistant", "content": "tool"}
        )
        provider.build_tool_result_messages = MagicMock(
            return_value=[{"role": "user", "content": "result"}]
        )

        tool_executor = MagicMock(spec=ToolExecutor)
        tool_executor.execute = MagicMock(
            return_value=ToolCall(
                tool_name="execute_python",
                tool_input={"code": "1"},
                output="1",
                is_error=False,
                duration=0.1,
            )
        )

        config = _make_config()
        config.max_tokens = 4096

        config.logs_dir = ""
        config.computation_token_alert = 999999
        config.progress_check_interval = 999

        with patch("open_dirac.llm._get_provider", return_value=provider):
            run_agent_loop(
                system="test_system",
                user_content="test",
                config=config,
                tool_executor=tool_executor,
                tools=[{"type": "function", "function": {"name": "execute_python"}}],
                max_rounds=max_rounds,
            )

        # The last call is the forced text-only call
        final_call = provider.call.call_args_list[-1]
        # System prompt should be UNCHANGED (no mutation)
        final_system = final_call.kwargs.get("system", "")
        assert final_system == "test_system"
        # The forced exit instruction is delivered as a user message
        final_messages = final_call.kwargs.get("messages", [])
        forced_user_msgs = [
            m
            for m in final_messages
            if isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and "You cannot call any more tools" in m["content"]
        ]
        assert len(forced_user_msgs) == 1
        assert "final output as text" in forced_user_msgs[0]["content"]


# ---------------------------------------------------------------------------
# HuggingFace _strip_tool_messages
# ---------------------------------------------------------------------------


class TestStripToolMessages:
    """Unit tests for HuggingFaceProvider._strip_tool_messages."""

    def test_removes_tool_role_messages(self):
        from open_dirac.providers.huggingface import HuggingFaceProvider

        msgs = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "I'll call a tool",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            {"role": "user", "content": "continue"},
        ]
        result = HuggingFaceProvider._strip_tool_messages(msgs)
        roles = [m["role"] for m in result]
        assert "tool" not in roles
        assert len(result) == 3

    def test_strips_tool_calls_key_from_assistant(self):
        from open_dirac.providers.huggingface import HuggingFaceProvider

        msgs = [
            {"role": "assistant", "content": "thinking", "tool_calls": [{"id": "tc1"}]},
        ]
        result = HuggingFaceProvider._strip_tool_messages(msgs)
        assert "tool_calls" not in result[0]
        assert result[0]["content"] == "thinking"

    def test_empty_content_gets_placeholder(self):
        from open_dirac.providers.huggingface import HuggingFaceProvider

        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
        ]
        result = HuggingFaceProvider._strip_tool_messages(msgs)
        assert result[0]["content"] == "[prior tool interaction omitted]"

    def test_passthrough_when_no_tools(self):
        from open_dirac.providers.huggingface import HuggingFaceProvider

        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = HuggingFaceProvider._strip_tool_messages(msgs)
        assert result == msgs
