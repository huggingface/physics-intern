"""Tests for vLLM provider — XML tool-call parsing, dual-mode handling, prompt rendering."""

from types import SimpleNamespace

from open_dirac.providers.vllm import VLLMProvider


# ---------------------------------------------------------------------------
# XML tool-call parsing
# ---------------------------------------------------------------------------


class TestParseXmlToolCalls:
    """Tests for VLLMProvider._parse_xml_tool_calls()."""

    def test_single_tool_call(self):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Paris</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = VLLMProvider._parse_xml_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["input"] == {"city": "Paris"}
        assert calls[0]["id"] == "xmlcall_0"

    def test_multiple_parameters(self):
        text = (
            "<tool_call>\n"
            "<function=multiply>\n"
            "<parameter=a>17</parameter>\n"
            "<parameter=b>42</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = VLLMProvider._parse_xml_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "multiply"
        assert calls[0]["input"] == {"a": 17, "b": 42}

    def test_multiple_tool_calls(self):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Paris</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=multiply>\n"
            "<parameter=a>2</parameter>\n"
            "<parameter=b>3</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = VLLMProvider._parse_xml_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["id"] == "xmlcall_0"
        assert calls[0]["name"] == "get_weather"
        assert calls[1]["id"] == "xmlcall_1"
        assert calls[1]["name"] == "multiply"

    def test_json_values_parsed(self):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=enabled>true</parameter>\n"
            "<parameter=count>5</parameter>\n"
            "<parameter=ratio>3.14</parameter>\n"
            "<parameter=label>hello world</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = VLLMProvider._parse_xml_tool_calls(text)
        assert calls[0]["input"]["enabled"] is True
        assert calls[0]["input"]["count"] == 5
        assert calls[0]["input"]["ratio"] == 3.14
        assert calls[0]["input"]["label"] == "hello world"

    def test_no_tool_calls(self):
        assert VLLMProvider._parse_xml_tool_calls("Just a regular response.") == []

    def test_empty_string(self):
        assert VLLMProvider._parse_xml_tool_calls("") == []

    def test_malformed_xml_no_function(self):
        text = "<tool_call>\nno function here\n</tool_call>"
        assert VLLMProvider._parse_xml_tool_calls(text) == []

    def test_surrounding_text_preserved(self):
        text = (
            "Let me check the weather.\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>London</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "I'll get back to you."
        )
        calls = VLLMProvider._parse_xml_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# Tool prompt rendering
# ---------------------------------------------------------------------------


class TestRenderToolsForPrompt:
    SAMPLE_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a math expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "e.g. '7 * 13'",
                        },
                    },
                    "required": ["expression"],
                },
            },
        },
    ]

    def test_contains_tool_name(self):
        rendered = VLLMProvider._render_tools_for_prompt(self.SAMPLE_TOOLS)
        assert "## calculate" in rendered

    def test_contains_description(self):
        rendered = VLLMProvider._render_tools_for_prompt(self.SAMPLE_TOOLS)
        assert "Evaluate a math expression." in rendered

    def test_contains_parameter_info(self):
        rendered = VLLMProvider._render_tools_for_prompt(self.SAMPLE_TOOLS)
        assert "expression" in rendered
        assert "(required)" in rendered

    def test_contains_xml_format_instructions(self):
        rendered = VLLMProvider._render_tools_for_prompt(self.SAMPLE_TOOLS)
        assert "<tool_call>" in rendered
        assert "<function=TOOL_NAME>" in rendered

    def test_multiple_tools(self):
        tools = self.SAMPLE_TOOLS + [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        rendered = VLLMProvider._render_tools_for_prompt(tools)
        assert "## calculate" in rendered
        assert "## search" in rendered


# ---------------------------------------------------------------------------
# Strip tool messages
# ---------------------------------------------------------------------------


class TestStripToolMessages:
    def test_removes_tool_role_messages(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ]
        cleaned = VLLMProvider._strip_tool_messages(messages)
        assert len(cleaned) == 2
        assert all(m["role"] != "tool" for m in cleaned)

    def test_strips_tool_calls_key(self):
        messages = [
            {"role": "assistant", "content": "text", "tool_calls": [{"id": "1"}]},
        ]
        cleaned = VLLMProvider._strip_tool_messages(messages)
        assert "tool_calls" not in cleaned[0]
        assert cleaned[0]["content"] == "text"

    def test_placeholder_when_no_content(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        ]
        cleaned = VLLMProvider._strip_tool_messages(messages)
        assert cleaned[0]["content"] == "[prior tool interaction omitted]"


# ---------------------------------------------------------------------------
# format_assistant_message
# ---------------------------------------------------------------------------


class TestFormatAssistantMessage:
    def _make_provider(self, tool_mode="api"):
        p = object.__new__(VLLMProvider)
        p._reasoning_format = ""
        p._tool_mode = tool_mode
        p._last_call_xml_tools = False
        return p

    def test_text_only(self):
        provider = self._make_provider()
        raw = SimpleNamespace(content="Hello!", tool_calls=None)
        msg = provider.format_assistant_message(raw)
        assert msg == {"role": "assistant", "content": "Hello!"}
        assert "tool_calls" not in msg

    def test_with_structured_tool_calls(self):
        provider = self._make_provider()
        raw = SimpleNamespace(
            content="",
            tool_calls=[
                SimpleNamespace(
                    id="call_0",
                    function=SimpleNamespace(
                        name="multiply",
                        arguments='{"a": 2, "b": 3}',
                    ),
                ),
            ],
        )
        msg = provider.format_assistant_message(raw)
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "multiply"

    def test_xml_mode_no_structured_tool_calls(self):
        """In xml_text mode, raw_content.tool_calls is None — message is text only."""
        provider = self._make_provider(tool_mode="xml_text")
        raw = SimpleNamespace(
            content="<tool_call><function=calc><parameter=x>1</parameter></function></tool_call>",
            tool_calls=None,
        )
        msg = provider.format_assistant_message(raw)
        assert "tool_calls" not in msg
        assert "<tool_call>" in msg["content"]


# ---------------------------------------------------------------------------
# build_tool_result_messages — dual mode
# ---------------------------------------------------------------------------


class TestBuildToolResultMessages:
    def _make_provider(self):
        p = object.__new__(VLLMProvider)
        p._reasoning_format = ""
        p._tool_mode = "api"
        return p

    def test_xml_mode_returns_user_messages(self):
        provider = self._make_provider()
        provider._last_call_xml_tools = True
        results = [
            {
                "tool_call_id": "xmlcall_0",
                "name": "get_weather",
                "output": '{"temp": 18}',
                "is_error": False,
            },
        ]
        msgs = provider.build_tool_result_messages(results)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "get_weather" in msgs[0]["content"]
        assert "result" in msgs[0]["content"]

    def test_xml_mode_multiple_results_combined(self):
        provider = self._make_provider()
        provider._last_call_xml_tools = True
        results = [
            {
                "tool_call_id": "xmlcall_0",
                "name": "get_weather",
                "output": "sunny",
                "is_error": False,
            },
            {
                "tool_call_id": "xmlcall_1",
                "name": "multiply",
                "output": "714",
                "is_error": False,
            },
        ]
        msgs = provider.build_tool_result_messages(results)
        assert len(msgs) == 1  # combined into one user message
        assert "get_weather" in msgs[0]["content"]
        assert "multiply" in msgs[0]["content"]

    def test_xml_mode_error_status(self):
        provider = self._make_provider()
        provider._last_call_xml_tools = True
        results = [
            {
                "tool_call_id": "xmlcall_0",
                "name": "execute_python",
                "output": "NameError: x not defined",
                "is_error": True,
            },
        ]
        msgs = provider.build_tool_result_messages(results)
        assert "error" in msgs[0]["content"]

    def test_api_mode_returns_tool_messages(self):
        provider = self._make_provider()
        provider._last_call_xml_tools = False
        results = [
            {
                "tool_call_id": "call_0",
                "name": "multiply",
                "output": "42",
                "is_error": False,
            },
        ]
        msgs = provider.build_tool_result_messages(results)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_0"
        assert msgs[0]["content"] == "42"


# ---------------------------------------------------------------------------
# prepare_messages — think tag stripping
# ---------------------------------------------------------------------------


class TestPrepareMessages:
    def _make_provider(self, reasoning_format="think_tags"):
        p = object.__new__(VLLMProvider)
        p._tool_mode = "api"
        p._last_call_xml_tools = False
        p._reasoning_format = reasoning_format
        return p

    def test_strips_think_from_older_turns(self):
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "question 1"},
            {"role": "assistant", "content": "<think>reasoning</think>Answer 1"},
            {"role": "user", "content": "question 2"},
            {"role": "assistant", "content": "<think>more reasoning</think>Answer 2"},
        ]
        result = provider.prepare_messages(messages)
        # Older assistant message should have think tags stripped
        assert "<think>" not in result[1]["content"]
        assert "Answer 1" in result[1]["content"]
        # Most recent assistant message should keep think tags
        assert "<think>" in result[3]["content"]

    def test_noop_without_think_tags_format(self):
        provider = self._make_provider(reasoning_format="")
        messages = [
            {"role": "assistant", "content": "<think>x</think>y"},
        ]
        result = provider.prepare_messages(messages)
        assert result[0]["content"] == "<think>x</think>y"

    def test_noop_no_assistant_messages(self):
        provider = self._make_provider()
        messages = [{"role": "user", "content": "hello"}]
        result = provider.prepare_messages(messages)
        assert result == messages
