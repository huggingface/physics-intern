"""OpenAI provider adapter (Responses API).

Uses /v1/responses unconditionally: /v1/chat/completions rejects function
tools combined with reasoning_effort on the gpt-5 family. We run stateless
(`store=False`) and pass `include=["reasoning.encrypted_content"]` so the
reasoning trace is preserved across tool round-trips via an opaque blob
echoed back in the next request — without relying on server-side state.
"""

import json
import os

from .base import LLMProvider, ProviderResponse

# Sentinel key used to carry replayed Responses output items through the
# shared messages list without colliding with role-based dicts.
_ITEMS_KEY = "_openai_items"


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (Responses API)."""

    def __init__(
        self,
        api_key: str = "",
        reasoning_effort: str = "",
        timeout: float = 600.0,
        **kwargs,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: uv sync --extra openai"
            )
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            timeout=timeout,
            # Disable the SDK's own retry loop; our wrapper in llm.py owns
            # retries so we can log them, cap total wall-time predictably,
            # and not multiply api_timeout by (1 + max_retries) per attempt.
            max_retries=0,
        )
        self._reasoning_effort = reasoning_effort

    def call(
        self,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ProviderResponse:
        kwargs = dict(
            model=model,
            instructions=system,
            input=self._messages_to_input(messages),
            max_output_tokens=max_tokens,
            store=False,
            include=["reasoning.encrypted_content"],
        )
        if tools:
            kwargs["tools"] = self._transform_tools(tools)
        if self._reasoning_effort:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}

        response = self._client.responses.create(**kwargs)
        return self._build_response(response)

    # ── Messages → Responses input ─────────────────────────────────────────

    @staticmethod
    def _messages_to_input(messages: list[dict]) -> list[dict]:
        """Flatten our mixed message list into a Responses input array.

        Entry shapes handled:
          - {"role": "user"|"assistant", "content": str | [...]}  — plain turn
          - {_ITEMS_KEY: [items...]}                              — replayed assistant items
          - {"type": "function_call_output", ...}                 — tool result
        """
        out: list[dict] = []
        for msg in messages:
            if _ITEMS_KEY in msg:
                out.extend(msg[_ITEMS_KEY])
                continue
            if msg.get("type") == "function_call_output":
                out.append(msg)
                continue
            role = msg.get("role")
            if role is None:
                out.append(msg)
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                part_type = "input_text" if role != "assistant" else "output_text"
                out.append(
                    {
                        "role": role,
                        "content": [{"type": part_type, "text": content}],
                    }
                )
            else:
                out.append({"role": role, "content": content})
        return out

    # ── Tool schema: chat-nested → responses-flat ─────────────────────────

    @staticmethod
    def _transform_tools(tools: list[dict]) -> list[dict]:
        result = []
        for t in tools:
            fn = t["function"]
            result.append(
                {
                    "type": "function",
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                    "strict": False,
                }
            )
        return result

    # ── Response parsing ───────────────────────────────────────────────────

    def _build_response(self, response) -> ProviderResponse:
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        replay_items: list[dict] = []

        for item in response.output:
            kind = getattr(item, "type", None)
            if kind == "message":
                for c in item.content:
                    if getattr(c, "type", None) == "output_text":
                        text_parts.append(c.text)
                replay_items.append(item.model_dump(exclude_none=True))
            elif kind == "function_call":
                args_str = item.arguments or ""
                try:
                    parsed_args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, ValueError):
                    parsed_args = {"raw": args_str}
                tool_calls.append(
                    {
                        "id": item.call_id,
                        "name": item.name,
                        "input": parsed_args,
                    }
                )
                replay_items.append(item.model_dump(exclude_none=True))
            elif kind == "reasoning":
                replay_items.append(item.model_dump(exclude_none=True))

        text = "".join(text_parts)

        if tool_calls:
            stop_reason = "tool_use"
        elif getattr(response, "status", None) == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", "") if details else ""
            stop_reason = "max_tokens" if reason == "max_output_tokens" else "end_turn"
        else:
            stop_reason = "end_turn"

        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        reasoning_tokens = 0
        details = getattr(usage, "output_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        answer_tokens = max(0, output_tokens - reasoning_tokens)

        return ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            tool_calls=tool_calls or None,
            raw_content=replay_items,
        )

    def format_assistant_message(self, raw_content: object) -> dict:
        # raw_content is the list of dumped output items (reasoning + message +
        # function_call). Wrap in a sentinel dict; _messages_to_input unpacks it.
        return {_ITEMS_KEY: raw_content}

    def build_tool_result_messages(self, tool_results: list[dict]) -> list[dict]:
        return [
            {
                "type": "function_call_output",
                "call_id": tr["tool_call_id"],
                "output": tr["output"],
            }
            for tr in tool_results
        ]
