"""Tests for reasoning token tracking across providers."""

from open_dirac.providers.base import (
    ProviderResponse,
    estimate_answer_tokens,
    estimate_reasoning_tokens,
    strip_think_tags,
)


# ── estimate_reasoning_tokens ──────────────────────────────────────────────


class TestEstimateReasoningTokens:
    def test_standard_think_tags(self):
        content = "Hello <think>Let me reason about this step by step</think> The answer is 4."
        tokens = estimate_reasoning_tokens(content)
        assert tokens > 0
        # 8 words * 1.3 = 10
        assert tokens == int(len("Let me reason about this step by step".split()) * 1.3)

    def test_qwen3_format_missing_opening_tag(self):
        """Qwen3 Thinking format: no <think>, just </think>."""
        content = "Let me think about this carefully</think>The answer is 42."
        tokens = estimate_reasoning_tokens(content)
        assert tokens > 0
        # Words before </think>
        assert tokens == int(len("Let me think about this carefully".split()) * 1.3)

    def test_empty_content(self):
        assert estimate_reasoning_tokens("") == 0
        assert estimate_reasoning_tokens(None) == 0

    def test_no_think_blocks(self):
        """Content without think tags returns 0."""
        assert estimate_reasoning_tokens("Just a normal response") == 0

    def test_empty_think_block(self):
        assert estimate_reasoning_tokens("<think></think>The answer") == 0

    def test_multiline_think_block(self):
        content = "<think>\nStep 1: Consider X\nStep 2: Apply Y\n</think>\nResult: Z"
        tokens = estimate_reasoning_tokens(content)
        assert tokens > 0


# ── estimate_answer_tokens ─────────────────────────────────────────────────


class TestEstimateAnswerTokens:
    def test_text_only(self):
        tokens = estimate_answer_tokens("The Hawking temperature is T")
        # 5 words * 1.3 = 6
        assert tokens == int(5 * 1.3)

    def test_empty_text(self):
        assert estimate_answer_tokens("") == 0
        assert estimate_answer_tokens("", None) == 0

    def test_tool_calls_only(self):
        """Tool-call-only response (text='') should still count tool args."""
        tool_calls = [{"id": "1", "name": "calc", "input": {"expr": "7*13"}}]
        tokens = estimate_answer_tokens("", tool_calls)
        assert tokens > 0

    def test_text_plus_tool_calls(self):
        tool_calls = [{"id": "1", "name": "calc", "input": {"expr": "7*13"}}]
        text_only = estimate_answer_tokens("The answer is 91")
        both = estimate_answer_tokens("The answer is 91", tool_calls)
        assert both > text_only

    def test_empty_tool_input(self):
        tool_calls = [{"id": "1", "name": "noop", "input": {}}]
        tokens = estimate_answer_tokens("", tool_calls)
        # json.dumps({}) = "{}" = 1 word * 1.3 = 1
        assert tokens >= 1

    def test_multiple_tool_calls(self):
        tool_calls = [
            {"id": "1", "name": "a", "input": {"x": 1}},
            {"id": "2", "name": "b", "input": {"y": "hello world"}},
        ]
        tokens = estimate_answer_tokens("", tool_calls)
        assert tokens > 0


# ── strip_think_tags ───────────────────────────────────────────────────────


class TestStripThinkTags:
    def test_standard_tags(self):
        text = "<think>Step 1: reason\nStep 2: more reasoning</think>The answer is 4."
        assert strip_think_tags(text) == "The answer is 4."

    def test_bare_closing_tag(self):
        """Qwen3/Nemotron: chat template inserts <think>, model emits only </think>."""
        text = "Let me think step by step...</think>The answer is 42."
        assert strip_think_tags(text) == "The answer is 42."

    def test_no_tags(self):
        text = "Just a normal response."
        assert strip_think_tags(text) == "Just a normal response."

    def test_empty(self):
        assert strip_think_tags("") == ""
        assert strip_think_tags(None) == ""

    def test_whitespace_only_think_block(self):
        text = "<think>   \n   </think>Result."
        assert strip_think_tags(text) == "Result."

    def test_multiple_think_blocks(self):
        text = "<think>first</think>mid<think>second</think>end"
        result = strip_think_tags(text)
        assert "first" not in result
        assert "second" not in result
        assert "mid" in result
        assert "end" in result


# ── ProviderResponse invariant ─────────────────────────────────────────────


class TestProviderResponseInvariant:
    def test_invariant_with_reasoning(self):
        """output_tokens == reasoning_tokens + answer_tokens when both set."""
        resp = ProviderResponse(
            text="answer",
            input_tokens=100,
            output_tokens=500,
            stop_reason="end_turn",
            reasoning_tokens=300,
            answer_tokens=200,
        )
        assert resp.output_tokens == resp.reasoning_tokens + resp.answer_tokens

    def test_defaults_zero(self):
        """New fields default to 0 for backward compatibility."""
        resp = ProviderResponse(
            text="answer",
            input_tokens=100,
            output_tokens=500,
            stop_reason="end_turn",
        )
        assert resp.reasoning_tokens == 0
        assert resp.answer_tokens == 0
        assert resp.reasoning_content == ""

    def test_reasoning_content_field(self):
        resp = ProviderResponse(
            text="answer",
            input_tokens=100,
            output_tokens=500,
            stop_reason="end_turn",
            reasoning_content="I thought about it carefully.",
        )
        assert resp.reasoning_content == "I thought about it carefully."

    def test_anthropic_paradigm(self):
        """Anthropic: output_tokens includes thinking, estimate split from text."""
        text = "The Hawking temperature is T equals h-bar kappa"
        output_tokens = 500
        answer_tokens = min(estimate_answer_tokens(text), output_tokens)
        reasoning_tokens = output_tokens - answer_tokens
        resp = ProviderResponse(
            text=text,
            input_tokens=100,
            output_tokens=output_tokens,
            stop_reason="end_turn",
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
        )
        assert resp.output_tokens == resp.reasoning_tokens + resp.answer_tokens

    def test_anthropic_tool_call_only(self):
        """Anthropic with text='' and tool_calls: answer_tokens should be nonzero."""
        tool_calls = [
            {
                "id": "1",
                "name": "execute_python",
                "input": {"code": "import numpy as np\nprint(np.pi)"},
            }
        ]
        output_tokens = 200
        answer_tokens = min(estimate_answer_tokens("", tool_calls), output_tokens)
        reasoning_tokens = output_tokens - answer_tokens
        assert answer_tokens > 0, "Tool call args should contribute to answer_tokens"
        assert reasoning_tokens + answer_tokens == output_tokens

    def test_openai_paradigm(self):
        """OpenAI: native reasoning_tokens from completion_tokens_details."""
        resp = ProviderResponse(
            text="4",
            input_tokens=50,
            output_tokens=200,
            stop_reason="end_turn",
            reasoning_tokens=180,
            answer_tokens=20,
        )
        assert resp.output_tokens == resp.reasoning_tokens + resp.answer_tokens

    def test_google_paradigm(self):
        """Google: answer = candidates_token_count, reasoning = thoughts_token_count."""
        answer = 100
        reasoning = 400
        resp = ProviderResponse(
            text="result",
            input_tokens=50,
            output_tokens=answer + reasoning,
            stop_reason="end_turn",
            reasoning_tokens=reasoning,
            answer_tokens=answer,
        )
        assert resp.output_tokens == resp.reasoning_tokens + resp.answer_tokens

    def test_huggingface_think_tags_paradigm(self):
        """HuggingFace with think_tags: estimate answer from visible text."""
        text = "<think>Step 1 step 2 step 3 reasoning</think>The answer is 4."
        output_tokens = 100
        visible_text = strip_think_tags(text)
        answer_tokens = min(estimate_answer_tokens(visible_text), output_tokens)
        reasoning_tokens = output_tokens - answer_tokens
        resp = ProviderResponse(
            text=text,
            input_tokens=50,
            output_tokens=output_tokens,
            stop_reason="end_turn",
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
        )
        assert resp.reasoning_tokens + resp.answer_tokens == resp.output_tokens

    def test_huggingface_separate_field_paradigm(self):
        """HuggingFace separate_field: estimate answer from text + tool calls."""
        output_tokens = 300
        tool_calls = [{"id": "1", "name": "calc", "input": {"expr": "7*13"}}]
        answer_tokens = min(
            estimate_answer_tokens("Let me calculate", tool_calls), output_tokens
        )
        reasoning_tokens = output_tokens - answer_tokens
        assert reasoning_tokens + answer_tokens == output_tokens

    def test_min_clamp_prevents_overshoot(self):
        """When estimate exceeds output_tokens, min() clamp preserves invariant."""
        text = "a " * 100  # 100 words → estimate ~130 tokens
        output_tokens = 50  # Less than estimate
        answer_tokens = min(estimate_answer_tokens(text), output_tokens)
        reasoning_tokens = output_tokens - answer_tokens
        assert answer_tokens == output_tokens
        assert reasoning_tokens == 0
        assert reasoning_tokens + answer_tokens == output_tokens


# ── LLMResponse / AgentResult fields ──────────────────────────────────────


class TestLLMWrapperFields:
    def test_llm_response_defaults(self):
        from open_dirac.llm import LLMResponse

        resp = LLMResponse(
            text="hi",
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
            duration=1.0,
        )
        assert resp.reasoning_tokens == 0
        assert resp.answer_tokens == 0

    def test_llm_response_with_reasoning(self):
        from open_dirac.llm import LLMResponse

        resp = LLMResponse(
            text="hi",
            input_tokens=10,
            output_tokens=50,
            stop_reason="end_turn",
            duration=1.0,
            reasoning_tokens=30,
            answer_tokens=20,
        )
        assert resp.reasoning_tokens == 30
        assert resp.answer_tokens == 20

    def test_agent_result_defaults(self):
        from open_dirac.llm import AgentResult

        result = AgentResult(text="done")
        assert result.total_reasoning_tokens == 0
        assert result.total_answer_tokens == 0

    def test_agent_result_with_reasoning(self):
        from open_dirac.llm import AgentResult

        result = AgentResult(
            text="done", total_reasoning_tokens=500, total_answer_tokens=200
        )
        assert result.total_reasoning_tokens == 500
        assert result.total_answer_tokens == 200
