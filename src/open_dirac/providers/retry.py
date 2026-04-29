"""Transient-error retry and error classification for LLM provider calls.

Pure, provider-agnostic. No sibling imports — observability is plumbed in via
the ``on_retry`` callback so higher layers can log to workspace/console without
dragging those dependencies into the provider layer.
"""

import re
import time
from collections.abc import Callable

from .base import LLMProvider, ProviderResponse


# ---------------------------------------------------------------------------
# Classifier constants
# ---------------------------------------------------------------------------

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524}

TRANSIENT_EXC_NAMES = {
    "ConnectionError",
    "TimeoutError",
    "ReadTimeout",
    "ConnectTimeout",
    "ConnectionResetError",
    "RemoteDisconnected",
    "BrokenPipeError",
    "APITimeoutError",
    "APIConnectionError",
    "APIStatusError",
    "ServerError",
    "RemoteProtocolError",
}

_PROVIDER_SIDE_400_PATTERNS = {
    "post processor",  # HuggingFace "gpt oss post processor" internal error
    "internal error",
    "backend error",
    "input validation",  # HuggingFace context-length / format rejection
}

_CONTEXT_TOO_LONG_PATTERNS = (
    "maximum context length",  # OpenAI / vLLM
    "prompt is too long",  # Anthropic
    "input is too long",  # HuggingFace
    "context window",  # generic
    "token limit",  # generic
    "reduce the length",  # OpenAI suggestion text
    "prompt_too_long",  # Google Gemini error code
    "context_length_exceeded",  # OpenAI error code
    "max_tokens",  # vLLM variants
)


# ---------------------------------------------------------------------------
# Exception type
# ---------------------------------------------------------------------------


class ContextTooLongError(Exception):
    """Raised when a provider rejects a request because the context is too long.

    Attributes:
        input_tokens:  Reported input token count (0 if not parseable).
        max_context:   Model's reported context limit (0 if not parseable).
        original:      The original provider exception.
    """

    def __init__(self, original: Exception):
        self.original = original
        self.input_tokens, self.max_context = parse_context_error(original)
        super().__init__(
            f"Context too long: ~{self.input_tokens} input tokens "
            f"(model limit {self.max_context}). "
            f"Original: {original}"
        )


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def extract_status_code(exc: Exception) -> int | None:
    """Extract a numeric HTTP status code from an exception, or None.

    Providers store status codes in various attributes and formats — e.g.
    google-genai sets .status to the string 'Bad Gateway' while .code holds
    the int 502.  We try several common attribute names and silently skip
    non-numeric values.
    """
    for attr in ("status_code", "code", "status"):
        val = getattr(exc, attr, None)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("status_code", "status"):
            val = getattr(resp, attr, None)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    continue
    return None


def is_tool_call_failure(exc: Exception) -> bool:
    """Return True if *exc* is a tool-call generation failure.

    Covers both JSON parse failures and OSS models ignoring tool_choice=none.
    """
    msg = str(exc).lower()
    return any(
        p in msg
        for p in (
            "tool_use_failed",
            "failed to parse tool call arguments",
            "output_parse_failed",  # HF backend can't parse non-tool output
            "tool choice",  # "Tool choice is none, but model called a tool"
        )
    )


def is_provider_side_400(exc: Exception) -> bool:
    """Return True if *exc* is an HTTP 400 from a provider-side processing failure.

    These are input-dependent (deterministic), not truly transient — retrying
    the identical request will almost always produce the same error.  We still
    allow a small number of retries (capped in call_with_retry) but give up
    early instead of burning all 10 attempts.
    """
    status = extract_status_code(exc)
    if status == 400:
        msg_lower = str(exc).lower()
        return any(p in msg_lower for p in _PROVIDER_SIDE_400_PATTERNS)
    return False


def is_context_too_long(exc: Exception) -> bool:
    """Return True if *exc* is a context-length / prompt-too-long rejection."""
    status = extract_status_code(exc)
    if status is not None and status not in (400, 413):
        return False
    msg_lower = str(exc).lower()
    return any(p in msg_lower for p in _CONTEXT_TOO_LONG_PATTERNS)


def parse_context_error(exc: Exception) -> tuple[int, int]:
    """Extract (input_tokens, max_context) from a context-length error message."""
    msg = str(exc)
    input_tokens = 0
    max_context = 0
    # OpenAI / vLLM: "contains at least 65537 input tokens"
    m = re.search(
        r"(?:contains|has)\s+(?:at\s+least\s+)?(\d+)\s+(?:input[_ ])?tokens", msg
    )
    if m:
        input_tokens = int(m.group(1))
    # OpenAI / vLLM: "maximum context length is 131072 tokens"
    m = re.search(r"maximum\s+context\s+length\s+(?:is|of)\s+(\d+)", msg)
    if m:
        max_context = int(m.group(1))
    return input_tokens, max_context


def is_transient(exc: Exception) -> bool:
    """Return True if *exc* looks like a transient / retryable API error."""
    # Context-too-long is deterministic — retrying wastes time and money
    if isinstance(exc, ContextTooLongError) or is_context_too_long(exc):
        return False
    # Tool-call generation failures are stochastic — retry may produce valid JSON
    if is_tool_call_failure(exc):
        return True
    # Check HTTP status code via robust extraction
    status = extract_status_code(exc)
    if status is not None and status in TRANSIENT_STATUS_CODES:
        return True
    # Provider-side 400s (post processor crashes, etc.) — retryable but capped
    if is_provider_side_400(exc):
        return True
    # Check exception type name anywhere in the MRO
    for cls in type(exc).__mro__:
        if cls.__name__ in TRANSIENT_EXC_NAMES:
            return True
    return False


# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------

RetryCallback = Callable[[Exception, int, int], None]
"""Signature: (exception, attempt_0_indexed, max_retries) -> None."""


def call_with_retry(
    provider: LLMProvider,
    *,
    max_retries: int,
    initial_delay: float,
    max_delay: float,
    on_retry: RetryCallback | None = None,
    **call_kwargs,
) -> ProviderResponse:
    """Retry ``provider.call(**call_kwargs)`` on transient errors with exponential backoff.

    - Context-too-long is deterministic: raised as ``ContextTooLongError`` immediately.
    - Provider-side 400s are capped at 1 retry (2 attempts total).
    - Other transient errors retry up to ``max_retries`` times with delay
      starting at ``initial_delay`` and doubling, capped at ``max_delay``.
    - ``on_retry(exc, attempt, max_retries)`` fires before each sleep so
      callers can plumb console/workspace logging without coupling this module.
    """
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return provider.call(**call_kwargs)
        except Exception as exc:
            if is_context_too_long(exc):
                raise ContextTooLongError(exc) from exc
            if not is_transient(exc) or attempt == max_retries:
                raise
            # Provider-side 400s are deterministic — cap at 1 retry (2 attempts total)
            if is_provider_side_400(exc) and attempt >= 1:
                raise
            if on_retry is not None:
                on_retry(exc, attempt, max_retries)
            time.sleep(min(delay, max_delay))
            delay *= 2
    raise RuntimeError("unreachable")  # pragma: no cover
