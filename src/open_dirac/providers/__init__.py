"""Provider factory and re-exports."""

from .base import (
    LLMProvider,
    ProviderResponse,
    estimate_answer_tokens,
    estimate_reasoning_tokens,
    strip_think_tags,
)
from .retry import (
    ContextTooLongError,
    call_with_retry,
    extract_status_code,
    is_context_too_long,
    is_provider_side_400,
    is_tool_call_failure,
    is_transient,
)


def create_provider(provider_name: str, api_key: str = "", **kwargs) -> LLMProvider:
    """Create a provider instance by name, with lazy imports."""
    if provider_name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(api_key=api_key, **kwargs)
    elif provider_name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(api_key=api_key, **kwargs)
    elif provider_name == "google":
        from .google import GoogleProvider

        return GoogleProvider(api_key=api_key, **kwargs)
    elif provider_name == "huggingface":
        from .huggingface import HuggingFaceProvider

        return HuggingFaceProvider(api_key=api_key, **kwargs)
    elif provider_name == "vllm":
        from .vllm import VLLMProvider

        return VLLMProvider(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


__all__ = [
    "LLMProvider",
    "ProviderResponse",
    "estimate_answer_tokens",
    "estimate_reasoning_tokens",
    "strip_think_tags",
    "create_provider",
    "ContextTooLongError",
    "call_with_retry",
    "extract_status_code",
    "is_context_too_long",
    "is_provider_side_400",
    "is_tool_call_failure",
    "is_transient",
]
