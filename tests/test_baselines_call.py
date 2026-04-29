"""Tests for open_dirac.baselines.call.run_baseline_call."""

from unittest.mock import MagicMock, patch

from open_dirac.baselines.call import run_baseline_call
from open_dirac.config import Config
from open_dirac.providers.base import ProviderResponse


def _make_config(**overrides):
    defaults = dict(
        api_retry_max=3,
        api_retry_initial_delay=0.01,
        api_retry_max_delay=0.1,
        progress_check_interval=999,
        input_cost=1.0,
        output_cost=2.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_response(text="ok", stop_reason="end_turn", input_tokens=10, output_tokens=5):
    return ProviderResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason=stop_reason,
    )


def test_returns_expected_dict_shape():
    """run_baseline_call wraps a single provider call and returns the
    structured dict both baselines rely on."""
    provider = MagicMock()
    provider.call.return_value = _make_response(text="hello")
    config = _make_config()

    result = run_baseline_call(
        provider,
        config,
        system="sys",
        user_message="q",
        agent_name="test",
    )

    assert set(result.keys()) == {
        "tokens",
        "duration_s",
        "cost_usd",
        "stop_reason",
        "response_text",
    }
    assert result["response_text"] == "hello"
    assert result["stop_reason"] == "end_turn"
    assert result["tokens"] == {
        "input": 10,
        "output": 5,
        "reasoning": 0,
        "answer": 0,
    }
    # input_cost=1.0, output_cost=2.0 per million ⇒ (10*1 + 5*2)/1e6 = 2e-5
    assert result["cost_usd"] == round(20 / 1_000_000, 6)


def test_invokes_max_tokens_continuation_when_truncated():
    """If the first response was truncated, continue_on_max_tokens is called."""
    provider = MagicMock()
    provider.call.return_value = _make_response(
        text="partial",
        stop_reason="max_tokens",
    )
    config = _make_config()

    cont_response = _make_response(
        text="partial and then more",
        stop_reason="end_turn",
        input_tokens=12,
        output_tokens=9,
    )
    with patch(
        "open_dirac.baselines.call.continue_on_max_tokens",
        return_value=cont_response,
    ) as mock_cont:
        result = run_baseline_call(
            provider,
            config,
            system="sys",
            user_message="q",
            agent_name="test",
        )

    assert mock_cont.call_count == 1
    assert result["stop_reason"] == "end_turn"
    assert result["response_text"] == "partial and then more"
