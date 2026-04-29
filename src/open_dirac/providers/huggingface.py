"""HuggingFace Inference Providers adapter."""

import json
import os
import re
from types import SimpleNamespace

from rich.console import Console

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
)

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference Providers via native InferenceClient."""

    # Back-compat alias — prefer importing ``strip_tool_messages`` from
    # ``._openai_compat`` directly.
    _strip_tool_messages = staticmethod(strip_tool_messages)

    def __init__(self, api_key: str = "", hf_provider: str | None = None,
                 timeout: float | None = None, reasoning_format: str = "",
                 **kwargs):
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "huggingface-hub package required. Install with: uv sync --extra huggingface"
            )
        token = api_key or os.environ.get("HF_TOKEN", "")
        client_kwargs = {"token": token}
        if hf_provider:
            client_kwargs["provider"] = hf_provider
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client = InferenceClient(**client_kwargs)
        self._reasoning_format = reasoning_format

    @staticmethod
    def _extract_failed_generation(exc: Exception) -> str:
        """Extract the failed_generation string from an HF error.

        Tries multiple strategies because the HF exception format varies:
        1. exc.response.json() → structured body with failed_generation key
        2. exc.response.text → raw JSON body parsed manually
        3. str(exc) → brace-counting extraction as last resort
        """
        # Strategy 1: structured response body via response.json()
        try:
            body = exc.response.json()  # type: ignore[union-attr]
            fg = body.get("failed_generation") or ""
            if not fg:
                err = body.get("error", {})
                if isinstance(err, dict):
                    fg = err.get("failed_generation") or ""
            if fg:
                return fg
        except Exception:
            pass

        # Strategy 2: parse raw response text
        try:
            raw_text = exc.response.text  # type: ignore[union-attr]
            body = json.loads(raw_text)
            fg = body.get("failed_generation") or ""
            if not fg:
                err = body.get("error", {})
                if isinstance(err, dict):
                    fg = err.get("failed_generation") or ""
            if fg:
                return fg
        except Exception:
            pass

        # Strategy 3: extract from str(exc) via brace-counting
        # HF includes the response body in the exception string.
        # Find the {"name": ...} tool-call object after "failed_generation".
        exc_str = str(exc)
        fg_idx = exc_str.find("failed_generation")
        if fg_idx < 0:
            return ""
        search_from = exc_str[fg_idx:]
        obj_start = search_from.find('{"name"')
        if obj_start < 0:
            return ""
        obj = search_from[obj_start:]
        # Count braces to find matching closing }
        depth = 0
        for i, ch in enumerate(obj):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return obj[: i + 1]
        # No matching brace — return everything (repair will strip trailing })
        return obj

    def _repair_failed_tool_call(self, exc: Exception) -> ProviderResponse | None:
        """Attempt to salvage a tool call from a tool_use_failed error.

        OSS models sometimes emit raw code instead of valid JSON for tool-call
        arguments, e.g. {"name": "execute_python", "arguments": import numpy ...}.
        The HF backend rejects this, but includes the raw generation in the error
        body.  We parse it out and build a synthetic ProviderResponse so the
        agentic loop can proceed instead of burning 10 identical retries.

        Returns None if repair is not possible.
        """
        console = Console(stderr=True)
        failed = self._extract_failed_generation(exc)
        if not failed:
            console.print(
                "[dim yellow]⚕ Repair: no failed_generation found[/]",
                highlight=False,
            )
            return None

        # Parse tool name
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', failed)
        if not name_match:
            return None
        tool_name = name_match.group(1)

        # Extract raw arguments after "arguments":
        arg_match = re.search(r'"arguments"\s*:\s*', failed)
        if not arg_match:
            return None
        raw_args = failed[arg_match.end():]
        # Strip a single trailing } that closes the outer tool-call object
        raw_args = raw_args.rstrip()
        if raw_args.endswith("}"):
            raw_args = raw_args[:-1].rstrip()

        # Try parsing as valid JSON first (backend rejected it for other reasons)
        try:
            parsed_args = json.loads(raw_args)
            if isinstance(parsed_args, str):
                # Quoted string — unwrap and wrap properly for execute_python
                parsed_args = {"code": parsed_args}
        except (json.JSONDecodeError, ValueError):
            # Raw code — wrap it for execute_python
            if tool_name == "execute_python":
                parsed_args = {"code": raw_args}
            else:
                return None

        console.print(
            f"[bold yellow]⚕ Repaired malformed tool call:[/] "
            f"{tool_name}({len(str(parsed_args))} chars)",
            highlight=False,
        )

        # Build synthetic raw_content compatible with format_assistant_message
        fake_tc = SimpleNamespace(
            id="repaired_0",
            function=SimpleNamespace(
                name=tool_name,
                arguments=json.dumps(parsed_args),
            ),
        )
        raw_content = SimpleNamespace(
            content="",
            tool_calls=[fake_tc],
        )

        return ProviderResponse(
            text="",
            input_tokens=0,
            output_tokens=0,
            stop_reason="tool_use",
            tool_calls=[{
                "id": "repaired_0",
                "name": tool_name,
                "input": parsed_args,
            }],
            raw_content=raw_content,
        )

    def call(self, model: str, max_tokens: int, system: str,
             messages: list[dict], tools: list[dict] | None = None) -> ProviderResponse:
        # HF uses OpenAI-compatible chat completions with system as first message
        hf_messages = [{"role": "system", "content": system}] + messages

        kwargs = dict(
            model=model,
            messages=hf_messages,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        if tools:
            kwargs["tools"] = tools  # Already in OpenAI canonical format
        else:
            # Strip tool-call history so OSS models don't hallucinate tool calls
            kwargs["messages"] = strip_tool_messages(hf_messages)

        try:
            stream = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            exc_msg = str(exc).lower()
            if "tool_use_failed" in exc_msg or "post processor" in exc_msg:
                repaired = self._repair_failed_tool_call(exc)
                if repaired is not None:
                    return repaired
            raise

        # Accumulate streamed chunks
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_acc: dict[int, dict] = {}  # index -> {id, name, arg_parts}
        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0

        for chunk in stream:
            # Usage info (typically in the final chunk)
            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            # Accumulate text content
            if hasattr(delta, "content") and delta.content:
                text_parts.append(delta.content)

            # Capture reasoning fields (GPT-OSS, Kimi, GLM)
            if hasattr(delta, "reasoning") and delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)

            # Accumulate tool call deltas
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
                            tc_acc[idx]["arg_parts"].append(
                                tc_delta.function.arguments)

        text = "".join(text_parts)

        # Guard against empty stream (no chunks received at all)
        if not text_parts and not tc_acc and finish_reason is None:
            raise RuntimeError(
                "Provider returned an empty stream (no chunks received). "
                "This may indicate a server-side timeout or misconfiguration."
            )

        # Build tool calls list + synthetic raw_content for format_assistant_message
        tool_calls = None
        raw_tool_calls = None
        if tc_acc:
            tool_calls = []
            raw_tool_calls = []
            for idx in sorted(tc_acc):
                tc = tc_acc[idx]
                args_str = "".join(tc["arg_parts"])
                tc_id = tc["id"] or f"call_{idx}"
                try:
                    parsed_args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    if tc["name"] == "execute_python":
                        parsed_args = {"code": args_str}
                    else:
                        parsed_args = {"raw": args_str}
                tool_calls.append({
                    "id": tc_id,
                    "name": tc["name"],
                    "input": parsed_args,
                })
                raw_tool_calls.append(build_raw_tool_call(tc_id, tc["name"], args_str))

        stop_reason = _STOP_REASON_MAP.get(finish_reason,
                                            finish_reason or "end_turn")

        # Synthetic raw_content (SimpleNamespace, same shape as non-streaming)
        raw_content = SimpleNamespace(
            content=text,
            tool_calls=raw_tool_calls,
        )

        # Determine effective reasoning format — auto-discover if not configured
        reasoning_content = "".join(reasoning_parts)
        fmt = self._reasoning_format
        if not fmt:
            if reasoning_content:
                fmt = "separate_field"
            elif "</think>" in text:
                fmt = "think_tags"

        # Reasoning token breakdown: always estimate answer first, derive reasoning
        reasoning_tokens = 0
        answer_tokens = output_tokens
        if fmt == "separate_field":
            # Models like Kimi, GPT-OSS: completion_tokens includes reasoning.
            reasoning_tokens, answer_tokens = split_reasoning_tokens(
                output_tokens, text, tool_calls)
        elif fmt == "think_tags":
            # Models like DeepSeek, Qwen: reasoning in <think>...</think> tags.
            # Extract reasoning content from tags, estimate answer from visible text.
            if not reasoning_content and "</think>" in text:
                match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
                if match:
                    reasoning_content = match.group(1)
                elif "</think>" in text:
                    reasoning_content = text.split("</think>", 1)[0]
            visible_text = strip_think_tags(text)
            reasoning_tokens, answer_tokens = split_reasoning_tokens(
                output_tokens, visible_text, tool_calls)

        return ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            tool_calls=tool_calls,
            raw_content=raw_content,
            reasoning_content=reasoning_content,
        )

    def format_assistant_message(self, raw_content: object) -> dict:
        msg = {"role": "assistant", "content": raw_content.content}
        if raw_content.tool_calls:
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
        """HF: separate role='tool' messages (OpenAI-compatible)."""
        messages = []
        for tr in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": tr["output"],
            })
        return messages
