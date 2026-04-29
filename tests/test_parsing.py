"""Tests for shared JSON parsing utilities in open_dirac.agents.parsing."""

import json

import pytest

from open_dirac.agents.parsing import (
    extract_json_with_error,
    fix_invalid_json_escapes,
    try_json_loads,
)


# ---------------------------------------------------------------------------
# fix_invalid_json_escapes
# ---------------------------------------------------------------------------


class TestFixInvalidJsonEscapes:
    def test_valid_double_backslash_preserved(self):
        r"""\\epsilon (already valid \\) should be preserved unchanged."""
        s = r'{"x": "\\epsilon"}'
        assert fix_invalid_json_escapes(s) == s

    def test_lone_backslash_doubled(self):
        r"""\( (lone backslash) should become \\(."""
        s = r'{"x": "\("}'
        fixed = fix_invalid_json_escapes(s)
        assert fixed == r'{"x": "\\("}'

    def test_mixed_escaping(self):
        r"""Mixed \(\\delta\) → \\(\\delta\\)."""
        s = r'{"x": "\(\\delta\)"}'
        fixed = fix_invalid_json_escapes(s)
        assert fixed == r'{"x": "\\(\\delta\\)"}'

    def test_valid_double_backslash_b(self):
        r"""\\bar — first letter 'b' is in valid set but \\b is valid JSON
        escape (backspace). The \\ pair should be preserved as-is."""
        s = r'{"x": "\\bar"}'
        assert fix_invalid_json_escapes(s) == s

    def test_valid_double_backslash_h(self):
        r"""\\hat — first letter 'h' NOT in valid set. But since \\ is a
        valid pair, it should be preserved unchanged (the regression case)."""
        s = r'{"x": "\\hat"}'
        assert fix_invalid_json_escapes(s) == s

    def test_standard_json_escapes_untouched(self):
        r"""Standard JSON escapes like \n, \t, \", \\ should pass through."""
        s = r'{"x": "line1\nline2\ttab\"quote\\"}'
        assert fix_invalid_json_escapes(s) == s

    def test_unicode_escape_untouched(self):
        r"""\u0041 should pass through unchanged."""
        s = r'{"x": "\u0041"}'
        assert fix_invalid_json_escapes(s) == s

    def test_latex_frac(self):
        r"""\frac should become \\frac (lone backslash, f not in valid set)."""
        s = r'{"x": "\frac{1}{2}"}'
        fixed = fix_invalid_json_escapes(s)
        # \f is a valid JSON escape (form feed), so it should NOT be doubled
        assert fixed == s

    def test_latex_delta(self):
        r"""\delta — lone backslash, d not in valid set → \\delta."""
        s = r'{"x": "\delta"}'
        fixed = fix_invalid_json_escapes(s)
        assert fixed == r'{"x": "\\delta"}'

    def test_no_backslashes(self):
        """String without backslashes should be unchanged."""
        s = '{"x": "hello world"}'
        assert fix_invalid_json_escapes(s) == s

    def test_malformed_unicode_u0delta(self):
        r"""\u0delta — malformed \uXXXX (not 4 hex digits) → \\u0delta."""
        s = r'{"x": "\u0delta"}'
        fixed = fix_invalid_json_escapes(s)
        assert fixed == r'{"x": "\\u0delta"}'

    def test_malformed_unicode_u2propto(self):
        r"""\u2propto — malformed \uXXXX → \\u2propto."""
        s = r'{"x": "\u2propto"}'
        fixed = fix_invalid_json_escapes(s)
        assert fixed == r'{"x": "\\u2propto"}'

    def test_valid_unicode_preserved(self):
        r"""\u03b2 (valid 4-hex unicode for β) should be unchanged."""
        s = r'{"x": "\u03b2"}'
        assert fix_invalid_json_escapes(s) == s

    def test_malformed_unicode_mixed_with_valid(self):
        r"""Mix of valid \u03b2 and malformed \u0delta."""
        s = r'{"x": "\u03b2 and \u0delta"}'
        fixed = fix_invalid_json_escapes(s)
        assert fixed == r'{"x": "\u03b2 and \\u0delta"}'


# ---------------------------------------------------------------------------
# try_json_loads
# ---------------------------------------------------------------------------


class TestTryJsonLoads:
    def test_valid_json_parses_directly(self):
        s = '{"result": "T_H = 1/(8*pi*M)", "confidence": "exact"}'
        parsed = try_json_loads(s)
        assert parsed["result"] == "T_H = 1/(8*pi*M)"

    def test_json_with_lone_backslash_fixed_and_parsed(self):
        r"""JSON with \( should be fixed and parsed."""
        s = r'{"result": "T_H = \(1/(8\pi M)\)", "confidence": "exact"}'
        parsed = try_json_loads(s)
        assert "T_H" in parsed["result"]
        assert parsed["confidence"] == "exact"

    def test_json_with_mixed_escaping_parsed(self):
        r"""JSON with mixed \\epsilon and \delta should be parsed."""
        s = r'{"result": "\\epsilon + \delta", "confidence": "approximate"}'
        parsed = try_json_loads(s)
        assert "epsilon" in parsed["result"]
        assert "delta" in parsed["result"]

    def test_truly_invalid_json_raises(self):
        """Truly invalid JSON (not just escape issues) should raise."""
        s = '{"result": broken}'
        with pytest.raises((json.JSONDecodeError, ValueError)):
            try_json_loads(s)

    def test_already_valid_json_with_backslash_n(self):
        """Valid JSON with \\n should parse without modification."""
        s = '{"x": "line1\\nline2"}'
        parsed = try_json_loads(s)
        assert parsed["x"] == "line1\nline2"

    def test_malformed_unicode_fixed_and_parsed(self):
        r"""JSON with malformed \u0delta should be fixed and parsed."""
        s = r'{"verdict": "VERIFIED", "details": "scaling \u0delta^{-4}"}'
        parsed = try_json_loads(s)
        assert parsed["verdict"] == "VERIFIED"
        assert "\\u0delta" in parsed["details"]

    def test_mixed_valid_and_malformed_unicode_parsed(self):
        r"""JSON with valid \u03b2 and malformed \u2propto should parse."""
        s = r'{"verdict": "VERIFIED", "details": "\u03b2 \u2propto"}'
        parsed = try_json_loads(s)
        assert parsed["verdict"] == "VERIFIED"
        assert "\u03b2" in parsed["details"]  # β character


# ---------------------------------------------------------------------------
# extract_json_with_error
# ---------------------------------------------------------------------------


class TestExtractJsonWithError:
    def test_valid_json_returns_parsed_and_no_error(self):
        text = '```json\n{"key": "value"}\n```'
        parsed, error = extract_json_with_error(text)
        assert parsed == {"key": "value"}
        assert error is None

    def test_valid_bare_json(self):
        text = 'Some preamble\n{"result": 42}\nsome trailing text'
        parsed, error = extract_json_with_error(text)
        assert parsed == {"result": 42}
        assert error is None

    def test_malformed_json_returns_error(self):
        text = '```json\n{"key": broken}\n```'
        parsed, error = extract_json_with_error(text)
        assert parsed is None
        assert error is not None
        assert "JSON parse error" in error

    def test_no_json_in_text(self):
        text = "This is just plain text with no JSON at all."
        parsed, error = extract_json_with_error(text)
        assert parsed is None
        assert error == "No JSON object found in response"

    def test_nemotron_string_concatenation_reports_error(self):
        r"""The \n    " pattern that breaks JSON array strings."""
        text = '{"sanity_checks": ["Check one.\\n    "Check two."]}'
        parsed, error = extract_json_with_error(text)
        assert parsed is None
        assert error is not None
        assert "JSON parse error" in error

    def test_fenced_preferred_over_bare(self):
        text = '{"bare": true}\n\n```json\n{"fenced": true}\n```'
        parsed, error = extract_json_with_error(text)
        assert parsed == {"fenced": True}
        assert error is None
