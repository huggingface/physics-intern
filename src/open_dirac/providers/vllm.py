"""vLLM local endpoint provider adapter.

Connects to a vLLM server via its OpenAI-compatible API.  Supports two
tool-calling modes, selected by the ``tool_mode`` config key in models.yaml:

``tool_mode: api`` (default)
    Standard OpenAI path — pass ``tools`` to the API, stream structured
    ``tool_calls`` deltas, return results as ``role: tool`` messages.
    Use for models served with ``--enable-auto-tool-choice``.

``tool_mode: xml_text``
    Text-based XML convention (e.g. Nemotron) — tool definitions are
    rendered as text in the system prompt, the model emits
    ``<tool_call>`` XML tags in its response, and results come back as
    user messages.  The ``tools`` parameter is NOT passed to the API
    (vLLM's tool-call interception layer interferes with the XML
    content in streaming mode).
"""

import json
import os
import re
from types import SimpleNamespace

from ._openai_compat import (
    build_raw_tool_call,
    canonical_tool_arguments_json_string,
    strip_tool_messages,
)
from .base import (
    LLMProvider,
    ProviderResponse,
    split_reasoning_tokens,
    strip_think_tags,
    transform_earlier_assistant_turns,
)

# ---------------------------------------------------------------------------
# XML tool-call regex patterns (Nemotron format)
# ---------------------------------------------------------------------------
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)
_FUNCTION_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>",
    re.DOTALL,
)

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


class VLLMProvider(LLMProvider):
    """vLLM OpenAI-compatible endpoint provider.

    Parameters
    ----------
    tool_mode : str
        ``"api"`` (default) for structured tool calls via the OpenAI API,
        ``"xml_text"`` for text-based XML tool calls (Nemotron style).
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "http://localhost:8000/v1",
        timeout: float = 600.0,
        reasoning_format: str = "",
        tool_mode: str = "api",
        **kwargs,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: uv sync --extra openai"
            )
        resolved_key = api_key or os.environ.get("VLLM_API_KEY", "") or "not-needed"
        # VLLM_BASE_URL env var overrides the base_url parameter from models.yaml,
        # allowing eval.slurm to point at the serve job's head node IP.
        resolved_url = os.environ.get("VLLM_BASE_URL", "") or base_url
        self._client = OpenAI(
            base_url=resolved_url,
            api_key=resolved_key,
            timeout=timeout,
        )
        self._reasoning_format = reasoning_format
        self._tool_mode = tool_mode
        # Tracks whether the most recent call() used xml_text tool mode.
        # Used by build_tool_result_messages() to choose message format.
        self._last_call_xml_tools: bool = False

    # ------------------------------------------------------------------
    # Tool description rendering (xml_text mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _render_tools_for_prompt(tools: list[dict]) -> str:
        """Convert OpenAI canonical tool definitions to a text description.

        The model sees this in the system prompt and responds with XML
        ``<tool_call>`` tags when it wants to invoke a tool.
        """
        lines = [
            "# Available Tools",
            "",
            "When you want to call a tool, output it in this exact XML format:",
            "",
            "<tool_call>",
            "<function=TOOL_NAME>",
            "<parameter=PARAM_NAME>value</parameter>",
            "</function>",
            "</tool_call>",
            "",
            "You may call multiple tools in one response, each in its own "
            "<tool_call> block.",
            "",
        ]
        for tool_def in tools:
            func = tool_def.get("function", tool_def)
            name = func["name"]
            desc = func.get("description", "")
            lines.append(f"## {name}")
            if desc:
                lines.append(desc)
            params = func.get("parameters", {})
            props = params.get("properties", {})
            required = set(params.get("required", []))
            if props:
                lines.append("Parameters:")
                for pname, pinfo in props.items():
                    ptype = pinfo.get("type", "string")
                    pdesc = pinfo.get("description", "")
                    req = " (required)" if pname in required else ""
                    line = f"- {pname} ({ptype}{req})"
                    if pdesc:
                        line += f": {pdesc}"
                    lines.append(line)
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # XML tool-call parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_xml_tool_calls(text: str) -> list[dict]:
        """Parse Nemotron XML tool calls from response text.

        Returns normalised tool calls: ``[{"id": ..., "name": ..., "input": {...}}]``
        or an empty list if none found / parse failure.
        """
        calls: list[dict] = []
        for idx, block_match in enumerate(_TOOL_CALL_RE.finditer(text)):
            block = block_match.group(1)
            func_match = _FUNCTION_RE.search(block)
            if not func_match:
                continue
            func_name = func_match.group(1)
            body = func_match.group(2)
            arguments: dict = {}
            for pm in _PARAMETER_RE.finditer(body):
                key = pm.group(1)
                raw_value = pm.group(2).strip()
                try:
                    arguments[key] = json.loads(raw_value)
                except (json.JSONDecodeError, ValueError):
                    arguments[key] = raw_value
            calls.append(
                {
                    "id": f"xmlcall_{idx}",
                    "name": func_name,
                    "input": arguments,
                }
            )
        return calls

    # Back-compat alias — prefer importing ``strip_tool_messages`` from
    # ``._openai_compat`` directly.
    _strip_tool_messages = staticmethod(strip_tool_messages)

    # ------------------------------------------------------------------
    # Core call — streaming
    # ------------------------------------------------------------------

    def call(
        self,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ProviderResponse:
        use_xml = self._tool_mode == "xml_text" and tools

        # Build system prompt — inject tool descriptions for xml_text mode
        effective_system = system
        if use_xml:
            effective_system = system + "\n\n" + self._render_tools_for_prompt(tools)

        vllm_messages = [{"role": "system", "content": effective_system}]
        if not tools:
            vllm_messages += strip_tool_messages(messages)
        else:
            vllm_messages += messages

        kwargs = dict(
            model=model,
            messages=vllm_messages,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        # api mode: pass tools to the API for structured tool calls
        if tools and not use_xml:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = self._client.chat.completions.create(**kwargs)

        # Accumulate streamed chunks
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_acc: dict[int, dict] = {}  # index -> {id, name, arg_parts}
        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0

        for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if hasattr(delta, "content") and delta.content:
                text_parts.append(delta.content)

            # vLLM reasoning parsers (e.g. nemotron_v3) return reasoning
            # in a separate field, stripped from content.
            # Field was renamed from "reasoning_content" to "reasoning"
            # in vLLM 0.19.0.
            rc = getattr(delta, "reasoning", None) or getattr(
                delta, "reasoning_content", None
            )
            if rc:
                reasoning_parts.append(rc)

            # Accumulate structured tool call deltas (api mode)
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_acc:
                        tc_acc[idx] = {"id": "", "name": "", "arg_parts": []}
                    if tc_delta.id:
                        tc_acc[idx]["id"] = tc_delta.id
                    if hasattr(tc_delta, "function") and tc_delta.function:
                        if tc_delta.function.name:
                            tc_acc[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_acc[idx]["arg_parts"].append(tc_delta.function.arguments)

        text = "".join(text_parts)

        # Guard against empty stream (no chunks received at all)
        if not text_parts and not tc_acc and finish_reason is None:
            raise RuntimeError(
                "Provider returned an empty stream (no chunks received). "
                "This may indicate a server-side timeout or misconfiguration."
            )

        # ------------------------------------------------------------------
        # Tool-call detection (mode-dependent)
        # ------------------------------------------------------------------
        tool_calls = None
        raw_tool_calls = None  # SimpleNamespace list for format_assistant_message

        if tc_acc:
            # Structured API tool calls (api mode)
            self._last_call_xml_tools = False
            tool_calls = []
            raw_tool_calls = []
            for idx in sorted(tc_acc):
                tc = tc_acc[idx]
                args_str = "".join(tc["arg_parts"])
                tc_id = tc["id"] or f"call_{idx}"
                try:
                    parsed_args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    parsed_args = {"raw": args_str}
                tool_calls.append(
                    {
                        "id": tc_id,
                        "name": tc["name"],
                        "input": parsed_args,
                    }
                )
                raw_tool_calls.append(build_raw_tool_call(tc_id, tc["name"], args_str))
        elif "<tool_call>" in text:
            # XML tool calls in text (xml_text mode)
            xml_calls = self._parse_xml_tool_calls(text)
            if xml_calls:
                self._last_call_xml_tools = True
                tool_calls = xml_calls
                raw_tool_calls = None
                finish_reason = "tool_calls"
        else:
            self._last_call_xml_tools = False

        stop_reason = _STOP_REASON_MAP.get(finish_reason, finish_reason or "end_turn")

        # raw_content for format_assistant_message
        raw_content = SimpleNamespace(
            content=text,
            tool_calls=raw_tool_calls,
        )

        # ------------------------------------------------------------------
        # Reasoning token breakdown
        # ------------------------------------------------------------------
        reasoning_content = ""
        # Server-side reasoning parser (e.g. nemotron_v3) already separated
        # reasoning from content — use it directly.
        if reasoning_parts:
            reasoning_content = "".join(reasoning_parts)

        fmt = self._reasoning_format
        if not fmt and "</think>" in text:
            fmt = "think_tags"

        reasoning_tokens = 0
        answer_tokens = output_tokens
        visible_text = text
        if reasoning_content and not re.search(r"<think>", text):
            # Reasoning was extracted server-side; text is already clean.
            reasoning_tokens, answer_tokens = split_reasoning_tokens(
                output_tokens, visible_text, tool_calls
            )
        elif fmt == "think_tags":
            match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            if match:
                reasoning_content = match.group(1)
            elif "</think>" in text:
                # Bare </think> — chat template inserts opening <think>
                reasoning_content = text.split("</think>", 1)[0]
            visible_text = strip_think_tags(text)
            reasoning_tokens, answer_tokens = split_reasoning_tokens(
                output_tokens, visible_text, tool_calls
            )

        return ProviderResponse(
            # text is the visible answer only; thinking trace lives in
            # reasoning_content and flows to LLMResponse.reasoning_content
            text=visible_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            tool_calls=tool_calls,
            raw_content=raw_content,
            reasoning_content=reasoning_content,
        )

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    def format_assistant_message(self, raw_content: object) -> dict:
        msg = {"role": "assistant", "content": raw_content.content}
        if raw_content.tool_calls:
            # Structured tool calls (api mode) — include in message
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": canonical_tool_arguments_json_string(
                            tc.function.arguments
                        ),
                    },
                }
                for tc in raw_content.tool_calls
            ]
        return msg

    def build_tool_result_messages(self, tool_results: list[dict]) -> list[dict]:
        """Build tool result messages.

        xml_text mode: user messages with text results (matches model training).
        api mode: standard role='tool' messages.
        """
        if self._last_call_xml_tools:
            parts = []
            for tr in tool_results:
                status = "error" if tr.get("is_error") else "result"
                parts.append(f"Tool '{tr['name']}' {status}: {tr['output']}")
            return [{"role": "user", "content": "\n\n".join(parts)}]
        else:
            messages = []
            for tr in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": tr["output"],
                    }
                )
            return messages

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Strip think tags from older assistant turns to save context."""
        if self._reasoning_format != "think_tags":
            return messages

        def _strip_tags(msg: dict) -> dict:
            if isinstance(msg.get("content"), str):
                return {**msg, "content": strip_think_tags(msg["content"])}
            return msg

        return transform_earlier_assistant_turns(messages, _strip_tags)
