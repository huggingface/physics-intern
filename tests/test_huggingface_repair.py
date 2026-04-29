"""Tests for HuggingFace provider malformed tool-call repair."""

import json
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeResponse:
    """Mimics httpx.Response with .json() and .text."""

    def __init__(self, body: dict):
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class BrokenJsonResponse:
    """Response where .json() fails but .text has valid JSON."""

    def __init__(self, body: dict):
        self.text = json.dumps(body)

    def json(self):
        raise ValueError("simulated json parse failure")


class FullyBrokenResponse:
    """Response where both .json() and .text-based parsing fail."""

    def __init__(self, exc_str_body: str):
        self._exc_str_body = exc_str_body
        self.text = "not json at all"

    def json(self):
        raise ValueError("json failed")


def _make_exc(failed_generation: str | None, *, nested: bool = False) -> Exception:
    """Build a fake exception with .response.json() carrying failed_generation."""
    if failed_generation is None:
        body = {"error": "something else"}
    elif nested:
        body = {
            "error": {
                "message": "tool_use_failed",
                "failed_generation": failed_generation,
            }
        }
    else:
        body = {"failed_generation": failed_generation}
    exc = Exception("tool_use_failed: bad tool call")
    exc.response = FakeResponse(body)
    return exc


def _make_exc_broken_json(failed_generation: str) -> Exception:
    """Build exception where .response.json() fails but .response.text works."""
    body = {"failed_generation": failed_generation}
    exc = Exception("tool_use_failed: bad tool call")
    exc.response = BrokenJsonResponse(body)
    return exc


def _make_exc_str_only(failed_generation: str) -> Exception:
    """Build exception where only str(exc) contains the failed_generation.

    Mimics real HF errors where the error body is embedded in the string repr.
    """
    # Simulate HfHubHTTPError string format:
    # "Bad request:\n{'message': '...', 'failed_generation': '...'}"
    msg = (
        f"(Request ID: req_test123)\n\n"
        f"Bad request:\n"
        f"{{'message': 'Tool choice is none', 'code': 'tool_use_failed', "
        f"'failed_generation': '{failed_generation}'}}"
    )
    exc = Exception(msg)
    exc.response = FullyBrokenResponse(msg)
    return exc


def _make_provider():
    """Create a HuggingFaceProvider with a mocked client."""
    with patch.dict("os.environ", {"HF_TOKEN": "fake"}):
        with patch("huggingface_hub.InferenceClient"):
            from open_dirac.providers.huggingface import HuggingFaceProvider

            return HuggingFaceProvider(api_key="fake")


# ---------------------------------------------------------------------------
# _extract_failed_generation unit tests
# ---------------------------------------------------------------------------


class TestExtractFailedGeneration:
    def test_strategy1_response_json(self):
        """Strategy 1: exc.response.json() with top-level failed_generation."""
        from open_dirac.providers.huggingface import HuggingFaceProvider

        raw = '{"name": "execute_python", "arguments": print(1)}'
        exc = _make_exc(raw)
        result = HuggingFaceProvider._extract_failed_generation(exc)
        assert result == raw

    def test_strategy1_nested(self):
        """Strategy 1: nested error dict."""
        from open_dirac.providers.huggingface import HuggingFaceProvider

        raw = '{"name": "execute_python", "arguments": x=1}'
        exc = _make_exc(raw, nested=True)
        result = HuggingFaceProvider._extract_failed_generation(exc)
        assert result == raw

    def test_strategy2_broken_json_method(self):
        """Strategy 2: .json() fails but .text has valid JSON."""
        from open_dirac.providers.huggingface import HuggingFaceProvider

        raw = '{"name": "execute_python", "arguments": print(42)}'
        exc = _make_exc_broken_json(raw)
        result = HuggingFaceProvider._extract_failed_generation(exc)
        assert result == raw

    def test_strategy3_str_extraction(self):
        """Strategy 3: both .json() and .text fail, extract from str(exc)."""
        from open_dirac.providers.huggingface import HuggingFaceProvider

        raw = '{"name": "execute_python", "arguments": import numpy as np}'
        exc = _make_exc_str_only(raw)
        result = HuggingFaceProvider._extract_failed_generation(exc)
        assert '"name": "execute_python"' in result
        assert "import numpy as np" in result

    def test_no_response_attr(self):
        """No .response at all → empty string."""
        from open_dirac.providers.huggingface import HuggingFaceProvider

        exc = Exception("some error without failed_generation")
        result = HuggingFaceProvider._extract_failed_generation(exc)
        assert result == ""

    def test_no_failed_generation_anywhere(self):
        """Nothing to extract → empty string."""
        from open_dirac.providers.huggingface import HuggingFaceProvider

        exc = _make_exc(None)
        result = HuggingFaceProvider._extract_failed_generation(exc)
        assert result == ""


# ---------------------------------------------------------------------------
# _repair_failed_tool_call unit tests
# ---------------------------------------------------------------------------


class TestRepairFailedToolCall:
    def test_raw_code_arguments(self):
        """Raw code (not JSON) → repaired as {"code": ...}."""
        provider = _make_provider()
        raw = (
            '{"name": "execute_python", "arguments": import numpy as np\nprint(np.pi)}'
        )
        exc = _make_exc(raw)
        result = provider._repair_failed_tool_call(exc)
        assert result is not None
        assert result.stop_reason == "tool_use"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["name"] == "execute_python"
        assert tc["input"] == {"code": "import numpy as np\nprint(np.pi)"}

    def test_quoted_string_arguments(self):
        """JSON string value → unwrapped and wrapped as {"code": ...}."""
        provider = _make_provider()
        raw = '{"name": "execute_python", "arguments": "print(42)"}'
        exc = _make_exc(raw)
        result = provider._repair_failed_tool_call(exc)
        assert result is not None
        assert result.tool_calls[0]["input"] == {"code": "print(42)"}

    def test_valid_json_arguments(self):
        """Valid JSON dict → passed through as-is."""
        provider = _make_provider()
        code = "import sympy\nprint(sympy.pi)"
        args_json = json.dumps({"code": code})
        raw = '{"name": "execute_python", "arguments": ' + args_json + "}"
        exc = _make_exc(raw)
        result = provider._repair_failed_tool_call(exc)
        assert result is not None
        assert result.tool_calls[0]["input"] == {"code": code}

    def test_no_failed_generation_returns_none(self):
        """Missing failed_generation → None."""
        provider = _make_provider()
        exc = _make_exc(None)
        assert provider._repair_failed_tool_call(exc) is None

    def test_no_response_attr_returns_none(self):
        """Exception without .response → None."""
        provider = _make_provider()
        exc = Exception("some other error")
        assert provider._repair_failed_tool_call(exc) is None

    def test_nested_error_dict(self):
        """failed_generation inside nested error dict → still found."""
        provider = _make_provider()
        raw = '{"name": "execute_python", "arguments": x = 42\nprint(x)}'
        exc = _make_exc(raw, nested=True)
        result = provider._repair_failed_tool_call(exc)
        assert result is not None
        assert result.tool_calls[0]["input"] == {"code": "x = 42\nprint(x)"}

    def test_multiline_code_with_special_chars(self):
        """Multiline code with quotes, braces, backslashes."""
        provider = _make_provider()
        code = 'x = {"a": 1}\nprint(f"result: {x}")\npath = "C:\\\\Users"'
        raw = '{"name": "execute_python", "arguments": ' + code + "}"
        exc = _make_exc(raw)
        result = provider._repair_failed_tool_call(exc)
        assert result is not None
        assert "execute_python" == result.tool_calls[0]["name"]
        assert "code" in result.tool_calls[0]["input"]

    def test_token_counts_zero(self):
        """Repaired responses have zero token counts."""
        provider = _make_provider()
        raw = '{"name": "execute_python", "arguments": print(1)}'
        exc = _make_exc(raw)
        result = provider._repair_failed_tool_call(exc)
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.reasoning_tokens == 0

    def test_unknown_tool_non_json_returns_none(self):
        """Non-JSON arguments for unknown tool (not execute_python) → None."""
        provider = _make_provider()
        raw = '{"name": "some_other_tool", "arguments": not json at all}'
        exc = _make_exc(raw)
        assert provider._repair_failed_tool_call(exc) is None

    def test_repair_via_strategy2(self):
        """Repair works when .json() fails but .text works."""
        provider = _make_provider()
        raw = '{"name": "execute_python", "arguments": print("strategy2")}'
        exc = _make_exc_broken_json(raw)
        result = provider._repair_failed_tool_call(exc)
        assert result is not None
        assert result.tool_calls[0]["input"] == {"code": 'print("strategy2")'}

    def test_repair_via_strategy3_str_only(self):
        """Repair works when only str(exc) has the failed_generation."""
        provider = _make_provider()
        raw = '{"name": "execute_python", "arguments": print("strategy3")}'
        exc = _make_exc_str_only(raw)
        result = provider._repair_failed_tool_call(exc)
        assert result is not None
        assert result.tool_calls[0]["name"] == "execute_python"
        assert "print" in result.tool_calls[0]["input"]["code"]

    def test_real_world_error_format(self):
        """Simulate exact error format from user's gpt-oss-120b output."""
        provider = _make_provider()
        # This is the exact format seen in production errors
        code = (
            "import numpy as np\n\n"
            "def analytic_mean_energy(x):\n"
            "    return 0.5 * (1/np.tanh(x/2))\n\n"
            "xs = [0.5, 5.0]\n"
            "for x in xs:\n"
            '    print(f"x={x:.3f}")\n'
        )
        raw = '{"name": "execute_python", "arguments": ' + code + "}"
        exc = _make_exc(raw)
        result = provider._repair_failed_tool_call(exc)
        assert result is not None
        assert result.tool_calls[0]["name"] == "execute_python"
        assert "analytic_mean_energy" in result.tool_calls[0]["input"]["code"]


# ---------------------------------------------------------------------------
# Integration: call() catches and repairs
# ---------------------------------------------------------------------------


class TestCallRepair:
    def test_call_returns_repaired_on_tool_use_failed(self):
        """call() catches tool_use_failed and returns repaired response."""
        provider = _make_provider()
        raw = '{"name": "execute_python", "arguments": print("hello")}'
        exc = _make_exc(raw)

        provider._client.chat.completions.create = MagicMock(side_effect=exc)

        tools = [{"type": "function", "function": {"name": "execute_python"}}]
        result = provider.call(
            model="test-model",
            max_tokens=4096,
            system="sys",
            messages=[],
            tools=tools,
        )
        assert result.stop_reason == "tool_use"
        assert result.tool_calls[0]["input"] == {"code": 'print("hello")'}

    def test_call_repairs_even_without_tools(self):
        """Repair works even for text-only calls (tools=None)."""
        provider = _make_provider()
        raw = '{"name": "execute_python", "arguments": print("text-only")}'
        exc = _make_exc(raw)
        provider._client.chat.completions.create = MagicMock(side_effect=exc)

        result = provider.call(
            model="test-model",
            max_tokens=4096,
            system="sys",
            messages=[],
            tools=None,
        )
        assert result.stop_reason == "tool_use"
        assert result.tool_calls[0]["input"] == {"code": 'print("text-only")'}

    def test_call_reraises_when_repair_fails(self):
        """If repair returns None, original exception is re-raised."""
        provider = _make_provider()
        exc = Exception("tool_use_failed")
        exc.response = FakeResponse({"error": "no failed_generation here"})
        provider._client.chat.completions.create = MagicMock(side_effect=exc)

        tools = [{"type": "function", "function": {"name": "execute_python"}}]
        with pytest.raises(Exception, match="tool_use_failed"):
            provider.call(
                model="test-model",
                max_tokens=4096,
                system="sys",
                messages=[],
                tools=tools,
            )

    def test_call_reraises_non_tool_use_error(self):
        """Non-tool_use_failed errors are never caught."""
        provider = _make_provider()
        exc = Exception("rate_limit_exceeded")
        provider._client.chat.completions.create = MagicMock(side_effect=exc)

        tools = [{"type": "function", "function": {"name": "execute_python"}}]
        with pytest.raises(Exception, match="rate_limit_exceeded"):
            provider.call(
                model="test-model",
                max_tokens=4096,
                system="sys",
                messages=[],
                tools=tools,
            )


# ---------------------------------------------------------------------------
# format_assistant_message compatibility
# ---------------------------------------------------------------------------


class TestFormatAssistantMessage:
    def test_repaired_raw_content_formats_correctly(self):
        """format_assistant_message works with SimpleNamespace raw_content."""
        provider = _make_provider()
        raw = '{"name": "execute_python", "arguments": print(99)}'
        exc = _make_exc(raw)
        result = provider._repair_failed_tool_call(exc)

        msg = provider.format_assistant_message(result.raw_content)
        assert msg["role"] == "assistant"
        assert msg["content"] == ""
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "execute_python"
        # arguments is a JSON string in the message format
        assert json.loads(tc["function"]["arguments"]) == {"code": "print(99)"}


# ---------------------------------------------------------------------------
# Post-processor error repair
# ---------------------------------------------------------------------------


def _make_post_processor_exc(failed_generation: str | None) -> Exception:
    """Build a fake 'gpt oss post processor' exception."""
    msg = "Encountered Exception during gpt oss post processor"
    if failed_generation is not None:
        body = {"error": msg, "failed_generation": failed_generation}
    else:
        body = {"error": msg}
    exc = Exception(msg)
    exc.status_code = 400
    exc.response = FakeResponse(body)
    return exc


class TestPostProcessorRepair:
    def test_repair_succeeds_with_failed_generation(self):
        """Post-processor error with failed_generation → repaired."""
        provider = _make_provider()
        raw = (
            '{"name": "execute_python", "arguments": import numpy as np\nprint(np.pi)}'
        )
        exc = _make_post_processor_exc(raw)
        result = provider._repair_failed_tool_call(exc)
        assert result is not None
        assert result.stop_reason == "tool_use"
        assert result.tool_calls[0]["name"] == "execute_python"
        assert "numpy" in result.tool_calls[0]["input"]["code"]

    def test_repair_fails_without_failed_generation(self):
        """Post-processor error without failed_generation → None."""
        provider = _make_provider()
        exc = _make_post_processor_exc(None)
        assert provider._repair_failed_tool_call(exc) is None

    def test_call_catches_post_processor_error(self):
        """call() catches 'post processor' errors and attempts repair."""
        provider = _make_provider()
        raw = '{"name": "execute_python", "arguments": print("post_proc")}'
        exc = _make_post_processor_exc(raw)
        provider._client.chat.completions.create = MagicMock(side_effect=exc)

        tools = [{"type": "function", "function": {"name": "execute_python"}}]
        result = provider.call(
            model="test-model",
            max_tokens=4096,
            system="sys",
            messages=[],
            tools=tools,
        )
        assert result.stop_reason == "tool_use"
        assert result.tool_calls[0]["input"] == {"code": 'print("post_proc")'}

    def test_call_reraises_post_processor_when_repair_fails(self):
        """Post-processor error without failed_generation → re-raised."""
        provider = _make_provider()
        exc = _make_post_processor_exc(None)
        provider._client.chat.completions.create = MagicMock(side_effect=exc)

        tools = [{"type": "function", "function": {"name": "execute_python"}}]
        with pytest.raises(Exception, match="post processor"):
            provider.call(
                model="test-model",
                max_tokens=4096,
                system="sys",
                messages=[],
                tools=tools,
            )
