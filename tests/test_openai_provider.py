"""Unit tests for OpenAI provider — request construction, Responses API kwargs."""

from types import SimpleNamespace

from open_dirac.providers.openai import OpenAIProvider


def _fake_response():
    """Minimal stand-in for a Responses API response object."""
    return SimpleNamespace(
        output=[],
        status="completed",
        usage=SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )


def _spy_client(captured: dict):
    """Build a provider whose responses.create records kwargs and returns a fake."""

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response()

    return SimpleNamespace(responses=SimpleNamespace(create=fake_create))


def _make_provider(reasoning_effort: str, captured: dict) -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = _spy_client(captured)
    provider._reasoning_effort = reasoning_effort
    return provider


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }
]


class TestReasoningEffortPropagation:
    """Regression: the provider must include reasoning.effort in the outgoing
    request whenever reasoning_effort is configured — including when tools
    are also present.  The original bug (commit 1170291) was an `elif` guard
    that dropped reasoning_effort on every tool-equipped call."""

    def test_reasoning_effort_sent_without_tools(self):
        captured: dict = {}
        provider = _make_provider("low", captured)
        provider.call(
            model="gpt-5.4",
            max_tokens=256,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert captured["reasoning"] == {"effort": "low"}

    def test_reasoning_effort_sent_with_tools(self):
        captured: dict = {}
        provider = _make_provider("low", captured)
        provider.call(
            model="gpt-5.4",
            max_tokens=256,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS,
        )
        assert captured["reasoning"] == {"effort": "low"}
        assert "tools" in captured

    def test_reasoning_absent_when_not_configured(self):
        captured: dict = {}
        provider = _make_provider("", captured)
        provider.call(
            model="gpt-5.4",
            max_tokens=256,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS,
        )
        assert "reasoning" not in captured


class TestResponsesApiKwargs:
    """The Responses API needs specific kwargs (instructions, input,
    max_output_tokens, store=False, include=[...]) — verify they're all set."""

    def test_core_kwargs(self):
        captured: dict = {}
        provider = _make_provider("", captured)
        provider.call(
            model="gpt-5.4",
            max_tokens=123,
            system="you are helpful",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert captured["model"] == "gpt-5.4"
        assert captured["instructions"] == "you are helpful"
        assert captured["max_output_tokens"] == 123
        assert captured["store"] is False
        assert captured["include"] == ["reasoning.encrypted_content"]

    def test_tools_transformed_to_flat_schema(self):
        captured: dict = {}
        provider = _make_provider("", captured)
        provider.call(
            model="gpt-5.4",
            max_tokens=256,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS,
        )
        tools = captured["tools"]
        assert len(tools) == 1
        # Responses API wants flat tool schema, not chat's nested {function: {...}}.
        assert tools[0]["type"] == "function"
        assert tools[0]["name"] == "add"
        assert "parameters" in tools[0]
        assert "function" not in tools[0]
