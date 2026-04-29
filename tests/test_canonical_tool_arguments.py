"""Tests for OpenAI-compat tool-call argument normalization."""

from __future__ import annotations

import json

from open_dirac.providers._openai_compat import canonical_tool_arguments_json_string


def test_empty_arguments_returns_empty_object() -> None:
    assert canonical_tool_arguments_json_string("") == "{}"
    assert canonical_tool_arguments_json_string(None) == "{}"
    assert canonical_tool_arguments_json_string("   ") == "{}"


def test_valid_json_is_compact_reserialized() -> None:
    raw = '  {"a": 1, "b": "x"}  '
    out = canonical_tool_arguments_json_string(raw)
    assert json.loads(out) == {"a": 1, "b": "x"}
    assert out == '{"a":1,"b":"x"}'


def test_invalid_json_becomes_diagnostic_object() -> None:
    broken = '{"code": "print(1)"'  # truncated
    out = canonical_tool_arguments_json_string(broken)
    data = json.loads(out)
    assert data["_open_dirac_tool_arguments_parse_error"] is True
    assert "parse_error_message" in data
    assert "snippet" in data
    assert broken.startswith(data["snippet"][:10])


def test_invalid_json_structure_yields_diagnostic() -> None:
    out = canonical_tool_arguments_json_string('{"a": }')
    data = json.loads(out)
    assert data["_open_dirac_tool_arguments_parse_error"] is True
