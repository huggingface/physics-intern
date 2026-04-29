"""Smoke tests — one real API call per provider.

These hit live APIs and cost real tokens, so they are skipped by default.
Run explicitly with:

    uv run python -m pytest tests/test_provider_smoke.py -v

Each test is skipped if the required API key is missing from the environment.
The prompt is trivial ("What is 2+2?") to minimise cost and latency.
"""

import os

import pytest

from open_dirac.providers import create_provider

PROMPT = "What is 2+2? Reply with just the number."
SYSTEM = "You are a helpful assistant. Be concise."
MAX_TOKENS = 256


def _simple_call(provider, model: str):
    """Fire one non-tool call and return the ProviderResponse."""
    return provider.call(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{"role": "user", "content": PROMPT}],
    )


def _assert_basic_response(resp):
    """Shared assertions for any valid smoke response."""
    assert resp.text, "Response text should not be empty"
    assert resp.stop_reason == "end_turn", f"Expected end_turn, got {resp.stop_reason}"
    assert resp.input_tokens > 0
    assert resp.output_tokens >= 0  # some providers report 0 with thinking enabled
    assert resp.reasoning_tokens >= 0
    assert resp.answer_tokens >= 0
    assert "4" in resp.text


# ── Anthropic ───────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestAnthropicSmoke:
    def test_basic_call(self):
        provider = create_provider("anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = _simple_call(provider, "claude-sonnet-4-6")
        _assert_basic_response(resp)

    def test_with_reasoning(self):
        provider = create_provider(
            "anthropic",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            reasoning_budget=1024,
        )
        resp = _simple_call(provider, "claude-sonnet-4-6")
        _assert_basic_response(resp)
        # With thinking enabled, reasoning_tokens should be positive
        assert resp.output_tokens == resp.reasoning_tokens + resp.answer_tokens


# ── OpenAI ──────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestOpenAISmoke:
    def test_basic_call(self):
        provider = create_provider("openai", api_key=os.environ["OPENAI_API_KEY"])
        resp = _simple_call(provider, "gpt-5.4")
        _assert_basic_response(resp)

    def test_with_reasoning(self):
        provider = create_provider(
            "openai",
            api_key=os.environ["OPENAI_API_KEY"],
            reasoning_effort="low",
        )
        resp = _simple_call(provider, "gpt-5.4")
        _assert_basic_response(resp)
        assert resp.output_tokens == resp.reasoning_tokens + resp.answer_tokens

    # Regression coverage for reasoning_effort being dropped when tools are
    # set lives in tests/test_openai_provider.py — that's a request-shape
    # check, not a model-behavior check, so it doesn't belong as a smoke test.


# ── Google ──────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)
class TestGoogleSmoke:
    def test_basic_call(self):
        provider = create_provider("google", api_key=os.environ["GOOGLE_API_KEY"])
        resp = _simple_call(provider, "gemini-3.1-pro-preview")
        _assert_basic_response(resp)

    def test_with_thinking(self):
        provider = create_provider(
            "google",
            api_key=os.environ["GOOGLE_API_KEY"],
            thinking_level="low",
        )
        resp = _simple_call(provider, "gemini-3.1-pro-preview")
        _assert_basic_response(resp)
        assert resp.output_tokens == resp.reasoning_tokens + resp.answer_tokens


# ── HuggingFace ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN not set",
)
class TestHuggingFaceSmoke:
    def test_basic_call(self):
        provider = create_provider("huggingface", api_key=os.environ["HF_TOKEN"])
        resp = _simple_call(provider, "Qwen/Qwen3.5-397B-A17B")
        _assert_basic_response(resp)
