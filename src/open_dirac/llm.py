"""LLM wrapper for OpenDirac — provider-agnostic via providers/ adapters."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .console import console
from .providers import (
    ContextTooLongError,
    LLMProvider,
    ProviderResponse,
    call_with_retry,
    create_provider,
    is_provider_side_400,
    is_tool_call_failure,
    strip_think_tags,
)
from .tool_call import ToolCall
from .agents.computer.tools import ToolExecutor
from .utils.categories import CompensationCategory as CC
from .workspace import log_llm_call, log_scaffold_event

_call_seq: dict[int, int] = {}
_round_num = round  # save builtin before parameter shadowing

# Provider cache: (provider_name, api_key) -> LLMProvider
_provider_cache: dict[tuple[str, str], LLMProvider] = {}


def _get_provider(config: Config) -> LLMProvider:
    """Create or retrieve a cached provider instance."""
    key = (config.provider, config.api_key)
    if key not in _provider_cache:
        _provider_cache[key] = create_provider(
            config.provider,
            api_key=config.api_key,
            timeout=config.api_timeout,
            **config.reasoning,
        )
    return _provider_cache[key]


class ParseFailureError(Exception):
    """Raised when an agent exhausts parse retries without valid output.

    Signals to the engine that the agent call failed to produce usable
    structured output and should be reported to the orchestrator — NOT
    stored as degraded partial evidence.
    """

    def __init__(self, agent_name: str, detail: str = ""):
        self.agent_name = agent_name
        self.detail = detail
        super().__init__(
            f"Parse failure after retries exhausted: agent={agent_name}"
            + (f", {detail}" if detail else "")
        )


def _call_provider_with_retry(
    provider: LLMProvider,
    config: Config,
    workspace_dir: str | Path = "",
    iteration: int = 0,
    **call_kwargs,
) -> ProviderResponse:
    """Retry provider.call() on transient errors with workspace-aware logging.

    Thin wrapper around ``providers.retry.call_with_retry`` that plumbs console
    and workspace observability. Retry policy itself lives in providers/retry.py.
    """

    def _on_retry(exc: Exception, attempt: int, max_retries: int) -> None:
        console.print(
            f"[yellow]Transient API error (attempt {attempt + 1}/"
            f"{max_retries}): {exc}[/yellow]"
        )
        if workspace_dir:
            log_scaffold_event(
                workspace_dir,
                iteration,
                CC.CALL_RELIABILITY,
                "api_retry",
                f"attempt={attempt + 1}/{max_retries}, {type(exc).__name__}",
            )

    return call_with_retry(
        provider,
        max_retries=config.api_retry_max,
        initial_delay=config.api_retry_initial_delay,
        max_delay=config.api_retry_max_delay,
        on_retry=_on_retry,
        **call_kwargs,
    )


# ---------------------------------------------------------------------------
# Max-tokens continuation
# ---------------------------------------------------------------------------

CONTINUATION_PROMPT = (
    "Your previous response was cut off because it hit the maximum output "
    "token limit. Continue from exactly where you stopped — do not repeat "
    "any text you have already written, resume at the next character."
)


def _merge_responses(
    first: ProviderResponse,
    cont: ProviderResponse,
) -> ProviderResponse:
    """Merge a first (truncated) response with a continuation response.

    text / reasoning_content / token counts are summed; stop_reason,
    tool_calls and raw_content come from the continuation (which carries
    the latest, non-truncated structured blocks). When the agentic loop
    persists the assistant turn via provider.format_assistant_message,
    we want the continuation's raw_content — not the truncated one.
    """
    return ProviderResponse(
        text=first.text + cont.text,
        input_tokens=first.input_tokens + cont.input_tokens,
        output_tokens=first.output_tokens + cont.output_tokens,
        stop_reason=cont.stop_reason,
        reasoning_tokens=first.reasoning_tokens + cont.reasoning_tokens,
        answer_tokens=first.answer_tokens + cont.answer_tokens,
        tool_calls=cont.tool_calls,
        raw_content=cont.raw_content,
        reasoning_content=(first.reasoning_content or "")
        + (cont.reasoning_content or ""),
    )


def continue_on_max_tokens(
    provider: LLMProvider,
    first_response: ProviderResponse,
    config: Config,
    *,
    model: str,
    max_tokens: int,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    workspace_dir: str | Path = "",
    iteration: int = 0,
    agent_name: str = "",
) -> ProviderResponse:
    """If *first_response* was truncated, attempt continuation(s).

    Sends back the visible answer portion of the truncated response (with
    ``<think>`` blocks stripped) as an assistant turn and asks the model to
    continue from where it stopped. Merged tokens are summed; stop_reason,
    tool_calls and raw_content come from the latest continuation attempt.

    Returns *first_response* unchanged when:
      - stop_reason is not "max_tokens" — nothing to continue.
      - The truncated response already produced ``tool_calls`` — partial
        tool JSON can't be safely continued.
      - The visible answer text is empty after ``strip_think_tags`` — the
        model spent its budget on reasoning without emitting an answer.
        Anthropic catches this one layer down via
        ``_call_with_thinking_recovery``; for Gemini / OpenAI reasoning
        models the CoT is opaque so we have nothing to send back; for OSS
        models feeding back an unclosed ``<think>`` is unsafe.

    If all ``config.max_tokens_retries`` attempts still end with
    ``stop_reason == "max_tokens"``, returns the merged (still truncated)
    response so downstream callers can log the unrecoverable truncation.
    """
    if first_response.stop_reason != "max_tokens":
        return first_response

    # Tool call already emitted — partial JSON, skip.
    if first_response.tool_calls:
        return first_response

    # Visible text check. Three ways a truncation can leave nothing useful
    # to feed back as the assistant turn:
    #   1. resp.text is empty or whitespace (OpenAI / Gemini reasoning-only).
    #   2. resp.text contains only complete <think>...</think> blocks —
    #      strip_think_tags leaves it empty.
    #   3. resp.text opened a <think> block that was never closed (truncated
    #      mid-reasoning for DeepSeek-style models) — strip_think_tags does
    #      NOT strip an unclosed block, so we detect it explicitly here.
    raw_text = first_response.text or ""
    visible = strip_think_tags(raw_text)
    truncated_mid_think = "<think>" in raw_text and "</think>" not in raw_text
    if not visible.strip() or truncated_mid_think:
        if workspace_dir:
            log_scaffold_event(
                workspace_dir,
                iteration,
                CC.LOOP_CONTROL,
                "max_tokens_reasoning_starved",
                f"agent={agent_name}, output_tokens={first_response.output_tokens}",
            )
        return first_response

    max_retries = config.max_tokens_retries
    if max_retries <= 0:
        return first_response

    merged = first_response
    for attempt in range(1, max_retries + 1):
        # Send back the visible-only portion of the accumulated response,
        # not raw_content — keeps the continuation turn clean across all
        # provider reasoning conventions (thinking blocks, thought parts,
        # inline <think> tags).
        cont_visible = strip_think_tags(merged.text or "")
        if not cont_visible.strip():
            # Continuation stripped to empty (all-reasoning retry) — give up.
            break
        cont_messages = list(messages) + [
            {"role": "assistant", "content": cont_visible},
            {"role": "user", "content": CONTINUATION_PROMPT},
        ]
        if workspace_dir:
            log_scaffold_event(
                workspace_dir,
                iteration,
                CC.LOOP_CONTROL,
                "max_tokens_continuation",
                f"agent={agent_name}, attempt={attempt}/{max_retries}",
            )
        console.print(
            f"[yellow]max_tokens reached — continuing response "
            f"(attempt {attempt}/{max_retries})[/yellow]"
        )
        call_kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=provider.prepare_messages(cont_messages),
        )
        if tools:
            call_kwargs["tools"] = tools
        try:
            cont_response = _call_provider_with_retry(
                provider,
                config,
                workspace_dir=workspace_dir,
                iteration=iteration,
                **call_kwargs,
            )
        except Exception as exc:
            console.print(
                f"[yellow]Continuation call failed: {type(exc).__name__}: {exc} "
                f"— returning best response so far[/yellow]"
            )
            return merged

        merged = _merge_responses(merged, cont_response)

        # Stop as soon as we get a clean finish (end_turn / tool_use).
        if cont_response.stop_reason != "max_tokens":
            return merged

    # All retries exhausted; merged still has stop_reason="max_tokens".
    if workspace_dir:
        log_scaffold_event(
            workspace_dir,
            iteration,
            CC.LOOP_CONTROL,
            "max_tokens_retries_exhausted",
            f"agent={agent_name}, attempts={max_retries}, "
            f"output_tokens={merged.output_tokens}",
        )
    return merged


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    duration: float
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    reasoning_content: str = (
        ""  # Parsed thinking trace (from <think> tags or separate field)
    )
    log_path: str = ""  # Path to conversation log file (set by call_llm)


@dataclass
class AgentResult:
    """Result from a multi-round tool-use agent loop."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    rounds: int = 0
    truncated: bool = False
    duration: float = 0.0
    stop_reason: str = "end_turn"
    token_alert_fired: bool = False
    total_reasoning_tokens: int = 0
    total_answer_tokens: int = 0


def call_llm(
    system: str,
    user_content: str,
    config: Config,
    agent_name: str = "",
    iteration: int = 0,
) -> LLMResponse:
    """Call the LLM with retry on transient errors."""
    if not user_content or not user_content.strip():
        raise ValueError(
            f"Empty user content in call_llm (agent={agent_name}, iteration={iteration})"
        )
    provider = _get_provider(config)

    # Write preliminary log (system + user) before the API call so the
    # prompt is preserved on disk even if the call fails.
    log_path = _allocate_log_path(config, agent_name, iteration)
    _write_preliminary_log(log_path, system, user_content)

    initial_messages = [{"role": "user", "content": user_content}]
    start = time.time()
    resp = _call_provider_with_retry(
        provider,
        config,
        workspace_dir=config.workspace_dir,
        iteration=iteration,
        model=config.model_id,
        max_tokens=config.max_tokens,
        system=system,
        messages=initial_messages,
    )
    # If the response was truncated, attempt continuation(s) in-place.
    if resp.stop_reason == "max_tokens":
        resp = continue_on_max_tokens(
            provider,
            resp,
            config,
            model=config.model_id,
            max_tokens=config.max_tokens,
            system=system,
            messages=initial_messages,
            workspace_dir=config.workspace_dir,
            iteration=iteration,
            agent_name=agent_name,
        )
    duration = time.time() - start

    llm_response = LLMResponse(
        text=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        stop_reason=resp.stop_reason,
        duration=duration,
        reasoning_tokens=resp.reasoning_tokens,
        answer_tokens=resp.answer_tokens,
        reasoning_content=resp.reasoning_content,
    )

    if config.workspace_dir:
        log_llm_call(
            config.workspace_dir,
            agent_name,
            iteration,
            config.model,
            llm_response.input_tokens,
            llm_response.output_tokens,
            llm_response.stop_reason,
            round(llm_response.duration, 2),
            len(system),
            len(user_content),
            len(llm_response.text),
            reasoning_tokens=llm_response.reasoning_tokens,
            answer_tokens=llm_response.answer_tokens,
        )

    # Overwrite preliminary log with full version (system + user + response)
    llm_response.log_path = _write_conversation_log(
        config,
        llm_response,
        system,
        user_content,
        agent_name,
        iteration,
        log_path=log_path,
    )

    return llm_response


def call_llm_continuation(
    system: str,
    messages: list[dict],
    config: Config,
    agent_name: str = "",
    iteration: int = 0,
    append_to_log: str = "",
) -> LLMResponse:
    """Continue a multi-turn conversation.

    Like :func:`call_llm` but accepts a full ``messages`` list (e.g.
    ``[user, assistant, user]``) and calls ``provider.prepare_messages()``
    so provider-specific context optimizations (thinking-block stripping,
    ``<think>`` tag removal) are applied to older turns.

    If *append_to_log* is a path to an existing log file, the retry turn
    (correction message + response) is appended there instead of creating
    a new file.
    """
    if not messages:
        raise ValueError(
            f"Empty messages in call_llm_continuation (agent={agent_name}, iteration={iteration})"
        )
    provider = _get_provider(config)
    prepared = provider.prepare_messages(list(messages))

    start = time.time()
    resp = _call_provider_with_retry(
        provider,
        config,
        workspace_dir=config.workspace_dir,
        iteration=iteration,
        model=config.model_id,
        max_tokens=config.max_tokens,
        system=system,
        messages=prepared,
    )
    if resp.stop_reason == "max_tokens":
        resp = continue_on_max_tokens(
            provider,
            resp,
            config,
            model=config.model_id,
            max_tokens=config.max_tokens,
            system=system,
            messages=list(messages),
            workspace_dir=config.workspace_dir,
            iteration=iteration,
            agent_name=agent_name,
        )
    duration = time.time() - start

    llm_response = LLMResponse(
        text=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        stop_reason=resp.stop_reason,
        duration=duration,
        reasoning_tokens=resp.reasoning_tokens,
        answer_tokens=resp.answer_tokens,
        reasoning_content=resp.reasoning_content,
    )

    if config.workspace_dir:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        log_llm_call(
            config.workspace_dir,
            agent_name,
            iteration,
            config.model,
            llm_response.input_tokens,
            llm_response.output_tokens,
            llm_response.stop_reason,
            _round_num(llm_response.duration, 2),
            len(system),
            len(last_user) if isinstance(last_user, str) else 0,
            len(llm_response.text),
            reasoning_tokens=llm_response.reasoning_tokens,
            answer_tokens=llm_response.answer_tokens,
        )

    # Append the retry turn to the original log file, or create a new one
    _append_retry_to_log(
        config, llm_response, messages, agent_name, iteration, append_to_log
    )

    return llm_response


def _append_retry_to_log(
    config: Config,
    resp: LLMResponse,
    messages: list[dict],
    agent_name: str,
    iteration: int,
    append_to_log: str,
):
    """Append the retry exchange (correction message + response) to a log file.

    If *append_to_log* points to an existing file, appends only the new
    turns (the last user message and the LLM response).  Otherwise falls
    back to creating a new standalone log file.
    """
    if not config.logs_dir:
        return

    # Extract the correction message (last user message in the conversation)
    correction = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    if not isinstance(correction, str):
        correction = str(correction)

    resp_attrs = (
        f'chars="{len(resp.text)}" tokens_in="{resp.input_tokens}" '
        f'tokens_out="{resp.output_tokens}"'
    )
    if resp.reasoning_tokens and resp.reasoning_tokens > 0:
        resp_attrs += (
            f' reasoning="{resp.reasoning_tokens}" answer="{resp.answer_tokens}"'
        )
    resp_attrs += f' duration="{resp.duration:.1f}s" stop="{resp.stop_reason}"'

    retry_section = (
        f'\n\n<RETRY_MESSAGE chars="{len(correction)}">\n'
        f"{correction}\n"
        f"</RETRY_MESSAGE>\n\n"
        f"<LLM_RESPONSE {resp_attrs}>\n"
        f"{resp.text}\n"
        f"</LLM_RESPONSE>\n"
    )

    # Try to append to existing log file
    if append_to_log:
        log_file = Path(append_to_log)
        if log_file.is_file():
            try:
                with log_file.open("a") as f:
                    f.write(retry_section)
                return
            except OSError:
                pass

    # Fallback: create a new log file with full context
    seq = _call_seq.get(iteration, 0) + 1
    _call_seq[iteration] = seq
    agent = agent_name or "unknown"
    filename = f"iter{iteration:03d}_{seq:02d}_{agent}.md"

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        tag = "USER_MESSAGE" if role == "USER" else "ASSISTANT_RESPONSE"
        parts.append(f'<{tag} chars="{len(content)}">\n{content}\n</{tag}>')
    parts.append(f"<LLM_RESPONSE {resp_attrs}>\n{resp.text}\n</LLM_RESPONSE>")

    try:
        Path(config.logs_dir, filename).write_text("\n\n".join(parts) + "\n")
    except OSError:
        pass


def run_agent_loop(
    system: str,
    user_content: str,
    config: Config,
    tool_executor: ToolExecutor,
    tools: list[dict],
    max_rounds: int = 10,
    agent_name: str = "",
    iteration: int = 0,
    on_round: Callable[[int, str, list[ToolCall], int, int, int, int, float], None]
    | None = None,
) -> AgentResult:
    """Run a tool-use agent loop until end_turn, max_rounds, or max_tokens."""
    provider = _get_provider(config)
    messages = [{"role": "user", "content": user_content}]

    all_tool_calls: list[ToolCall] = []
    round_log: list[dict] = []

    # Pre-compute log filepath for progressive writing (increment seq exactly once)
    _log_filepath: Path | None = None
    if config.logs_dir:
        _log_seq = _call_seq.get(iteration, 0) + 1
        _call_seq[iteration] = _log_seq
        _log_agent = agent_name or "unknown"
        _log_filepath = Path(
            config.logs_dir, f"iter{iteration:03d}_{_log_seq:02d}_{_log_agent}.md"
        )

    def _flush_log() -> None:
        """Rewrite the log file with current accumulated state."""
        if _log_filepath is None:
            return
        content = _render_agent_conversation_log(system, user_content, round_log, tools)
        try:
            _log_filepath.write_text(content)
        except OSError:
            pass

    # Write preamble (system + tools + user, no rounds yet)
    _flush_log()

    total_input = 0
    total_output = 0
    total_reasoning = 0
    total_answer = 0
    consecutive_exec_python = 0
    token_alert_fired = False
    overall_start = time.time()

    tool_call_failure = False
    empty_end_turn_count = 0
    ready_conclude_recovery_count = 0
    text_end_turn_recovery_count = 0
    loop_exit_reason = "max_rounds"  # default; overridden on early exits
    # Resolve exit tool name(s) for context-aware messages
    _exit_tool_names: frozenset[str] = getattr(
        tool_executor,
        "exit_tool_names",
        frozenset({getattr(tool_executor, "exit_tool_name", "submit_review")}),
    )
    if len(_exit_tool_names) == 1:
        _exit_tool_msg = f"`{next(iter(_exit_tool_names))}`"
    else:
        _exit_tool_msg = (
            "a dispatch tool ("
            + ", ".join(f"`{n}`" for n in sorted(_exit_tool_names))
            + ")"
        )
    for round_num in range(1, max_rounds + 1):
        start = time.time()
        # Allow executor to dynamically switch tool set (e.g. after document_approach)
        active_tools = getattr(tool_executor, "active_tools", None) or tools
        # Flush accumulated log state before each API call so the prompt
        # is on disk even if the call fails (mirrors preliminary logging
        # in call_llm).
        _flush_log()
        try:
            resp = _call_provider_with_retry(
                provider,
                config,
                workspace_dir=config.workspace_dir,
                iteration=iteration,
                model=config.model_id,
                max_tokens=config.max_tokens,
                system=system,
                messages=provider.prepare_messages(messages),
                tools=active_tools,
            )
        except ContextTooLongError as exc:
            console.print(
                f"[yellow]Context too long (round {round_num}): {exc} "
                f"— falling back to text-only response[/yellow]"
            )
            tool_call_failure = True
            loop_exit_reason = "context_too_long"
            if config.workspace_dir:
                log_scaffold_event(
                    config.workspace_dir,
                    iteration,
                    CC.CALL_RELIABILITY,
                    "context_too_long_fallback",
                    f"round={round_num}, input_tokens={exc.input_tokens}",
                )
            break
        except Exception as exc:
            if is_tool_call_failure(exc):
                console.print(
                    f"[yellow]Tool-call generation failed after retries "
                    f"(round {round_num}): {exc} — falling back to "
                    f"text-only response[/yellow]"
                )
                tool_call_failure = True
                loop_exit_reason = "tool_call_failure"
                if config.workspace_dir:
                    log_scaffold_event(
                        config.workspace_dir,
                        iteration,
                        CC.CALL_RELIABILITY,
                        "tool_call_failure_fallback",
                        f"round={round_num}",
                    )
                break
            if is_provider_side_400(exc):
                console.print(
                    f"[yellow]Provider-side 400 after retries "
                    f"(round {round_num}): {exc} — falling back to "
                    f"text-only response[/yellow]"
                )
                tool_call_failure = True
                loop_exit_reason = "provider_side_400"
                if config.workspace_dir:
                    log_scaffold_event(
                        config.workspace_dir,
                        iteration,
                        CC.CALL_RELIABILITY,
                        "provider_side_400_fallback",
                        f"round={round_num}",
                    )
                break
            raise
        # If the round was truncated, try to continue in place before
        # falling through to the max_tokens exit branch below. When the
        # continuation completes cleanly (end_turn / tool_use), the round
        # proceeds normally with the merged response.
        if resp.stop_reason == "max_tokens":
            resp = continue_on_max_tokens(
                provider,
                resp,
                config,
                model=config.model_id,
                max_tokens=config.max_tokens,
                system=system,
                messages=messages,
                tools=active_tools,
                workspace_dir=config.workspace_dir,
                iteration=iteration,
                agent_name=agent_name,
            )
        duration = time.time() - start

        round_log.append(
            {
                "kind": "llm_response",
                "round": round_num,
                "text": resp.text,
                "tool_calls": resp.tool_calls,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "reasoning_tokens": resp.reasoning_tokens,
                "answer_tokens": resp.answer_tokens,
                "stop_reason": resp.stop_reason,
                "duration": duration,
            }
        )

        total_input += resp.input_tokens
        total_output += resp.output_tokens
        total_reasoning += resp.reasoning_tokens
        total_answer += resp.answer_tokens
        if total_input > config.computation_token_alert:
            token_alert_fired = True

        # Audit + conversation log for this round
        round_text = resp.text
        round_resp = LLMResponse(
            text=round_text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            stop_reason=resp.stop_reason,
            duration=duration,
            reasoning_tokens=resp.reasoning_tokens,
            answer_tokens=resp.answer_tokens,
        )
        if config.workspace_dir:
            log_llm_call(
                config.workspace_dir,
                agent_name,
                iteration,
                config.model,
                round_resp.input_tokens,
                round_resp.output_tokens,
                round_resp.stop_reason,
                _round_num(round_resp.duration, 2),
                len(system),
                len(user_content),
                len(round_resp.text),
                reasoning_tokens=round_resp.reasoning_tokens,
                answer_tokens=round_resp.answer_tokens,
                round=round_num,
            )

        # end_turn: done (unless empty text after tool calls — attempt recovery)
        if resp.stop_reason == "end_turn":
            if not round_text.strip() and all_tool_calls:
                empty_end_turn_count += 1
                # Inject recovery message and retry — the for loop
                # naturally exhausts at max_rounds
                _recovery_msg = (
                    f"Your response was empty. You still need to call {_exit_tool_msg} "
                    f"to complete your task. If you are done, review your findings and call "
                    f"{_exit_tool_msg} now."
                )
                messages.append(provider.format_assistant_message(resp.raw_content))
                messages.append({"role": "user", "content": _recovery_msg})
                round_log.append(
                    {
                        "kind": "scaffold_injection",
                        "round": round_num,
                        "label": "empty_end_turn_recovery",
                        "content": _recovery_msg,
                    }
                )
                if config.workspace_dir:
                    log_scaffold_event(
                        config.workspace_dir,
                        iteration,
                        CC.CALL_RELIABILITY,
                        "empty_end_turn_recovery",
                        f"round={round_num}, count={empty_end_turn_count}",
                    )
                continue
            empty_end_turn_count = 0  # non-empty turn resets counter

            # Ready-to-conclude recovery: agent signaled ready but answered in text
            if getattr(tool_executor, "ready_to_conclude_signaled", False):
                ready_conclude_recovery_count += 1
                if ready_conclude_recovery_count == 1:
                    _recovery_msg = (
                        f"You produced output but did not call {_exit_tool_msg}. "
                        f"You reported ready to conclude — call {_exit_tool_msg} now "
                        f"with your findings."
                    )
                    messages.append(provider.format_assistant_message(resp.raw_content))
                    messages.append({"role": "user", "content": _recovery_msg})
                    round_log.append(
                        {
                            "kind": "scaffold_injection",
                            "round": round_num,
                            "label": "ready_conclude_recovery",
                            "content": _recovery_msg,
                        }
                    )
                    if config.workspace_dir:
                        log_scaffold_event(
                            config.workspace_dir,
                            iteration,
                            CC.CALL_RELIABILITY,
                            "ready_conclude_recovery",
                            f"round={round_num}",
                        )
                    continue
                # Second time: fall through to normal end_turn return

            # Text end_turn recovery: model wrote findings as text without calling exit tool
            if (
                round_text.strip()
                and all_tool_calls
                and not getattr(tool_executor, "stop_after_round", False)
                and not getattr(tool_executor, "ready_to_conclude_signaled", False)
            ):
                text_end_turn_recovery_count += 1
                if text_end_turn_recovery_count == 1:
                    _recovery_msg = (
                        f"You wrote your findings as text but did not call {_exit_tool_msg}. "
                        f"You MUST call {_exit_tool_msg} with your results to complete your task. "
                        f"Use the text you just wrote to fill in the {_exit_tool_msg} parameters."
                    )
                    messages.append(provider.format_assistant_message(resp.raw_content))
                    messages.append({"role": "user", "content": _recovery_msg})
                    round_log.append(
                        {
                            "kind": "scaffold_injection",
                            "round": round_num,
                            "label": "text_end_turn_recovery",
                            "content": _recovery_msg,
                        }
                    )
                    if config.workspace_dir:
                        log_scaffold_event(
                            config.workspace_dir,
                            iteration,
                            CC.CALL_RELIABILITY,
                            "text_end_turn_recovery",
                            f"round={round_num}",
                        )
                    continue
                # Second time: fall through to normal end_turn return

            result = AgentResult(
                text=round_text,
                tool_calls=all_tool_calls,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                rounds=round_num,
                truncated=False,
                duration=time.time() - overall_start,
                stop_reason="end_turn",
                token_alert_fired=token_alert_fired,
                total_reasoning_tokens=total_reasoning,
                total_answer_tokens=total_answer,
            )
            _write_agent_conversation_log(
                config,
                system,
                user_content,
                agent_name,
                iteration,
                round_log,
                result,
                tools=tools,
                _filepath=_log_filepath,
            )
            return result

        # max_tokens: truncated
        if resp.stop_reason == "max_tokens":
            result = AgentResult(
                text=round_text,
                tool_calls=all_tool_calls,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                rounds=round_num,
                truncated=True,
                duration=time.time() - overall_start,
                stop_reason="max_tokens",
                token_alert_fired=token_alert_fired,
                total_reasoning_tokens=total_reasoning,
                total_answer_tokens=total_answer,
            )
            _write_agent_conversation_log(
                config,
                system,
                user_content,
                agent_name,
                iteration,
                round_log,
                result,
                tools=tools,
                _filepath=_log_filepath,
            )
            return result

        # tool_use: execute tools and continue
        if resp.stop_reason == "tool_use":
            messages.append(provider.format_assistant_message(resp.raw_content))

            if hasattr(tool_executor, "begin_round"):
                tool_executor.begin_round()

            tool_results = []
            for tc_info in resp.tool_calls:
                tc = tool_executor.execute(tc_info["name"], tc_info["input"])
                all_tool_calls.append(tc)
                round_log.append(
                    {
                        "kind": "tool_result",
                        "round": round_num,
                        "tool_name": tc.tool_name,
                        "tool_input": tc.tool_input,
                        "output": tc.output,
                        "is_error": tc.is_error,
                        "duration": tc.duration,
                    }
                )
                tool_results.append(
                    {
                        "tool_call_id": tc_info["id"],
                        "name": tc_info["name"],
                        "output": tc.output,
                        "is_error": tc.is_error,
                    }
                )

            messages.extend(provider.build_tool_result_messages(tool_results))

            # State injection AFTER tool results so the model sees it
            if hasattr(tool_executor, "end_round"):
                injection = tool_executor.end_round()
                if injection:
                    messages.append({"role": "user", "content": injection})
                    round_log.append(
                        {
                            "kind": "scaffold_injection",
                            "round": round_num,
                            "label": "state_injection",
                            "content": injection,
                        }
                    )

            if config.workspace_dir:
                for tc in all_tool_calls[-len(tool_results) :]:
                    if tc.output.startswith("TIMEOUT:"):
                        log_scaffold_event(
                            config.workspace_dir,
                            iteration,
                            CC.CALL_RELIABILITY,
                            "tool_timeout",
                            f"round={round_num}",
                        )
                    elif "[... truncated" in tc.output or "[...truncated" in tc.output:
                        log_scaffold_event(
                            config.workspace_dir,
                            iteration,
                            CC.CALL_RELIABILITY,
                            "tool_output_truncation",
                            f"round={round_num}",
                        )

            # Notify caller about round progress
            round_tool_calls = [tc for tc in all_tool_calls[-len(tool_results) :]]
            if on_round:
                on_round(
                    round_num,
                    resp.stop_reason,
                    round_tool_calls,
                    total_input,
                    total_output,
                    resp.input_tokens,
                    resp.output_tokens,
                    round_duration=duration,
                    round_reasoning=resp.reasoning_tokens,
                    round_answer=resp.answer_tokens,
                )

            # Executor-signaled early stop (e.g., orchestrator dispatch tools)
            if getattr(tool_executor, "stop_after_round", False):
                result = AgentResult(
                    text=round_text,
                    tool_calls=all_tool_calls,
                    total_input_tokens=total_input,
                    total_output_tokens=total_output,
                    total_reasoning_tokens=total_reasoning,
                    total_answer_tokens=total_answer,
                    rounds=round_num,
                    truncated=False,
                    stop_reason="executor_stop",
                    duration=time.time() - overall_start,
                    token_alert_fired=token_alert_fired,
                )
                _write_agent_conversation_log(
                    config,
                    system,
                    user_content,
                    agent_name,
                    iteration,
                    round_log,
                    result,
                    tools=tools,
                    _filepath=_log_filepath,
                )
                return result

            # Track consecutive execute_python for progress check
            round_has_exec = any(
                tc.tool_name == "execute_python"
                for tc in all_tool_calls[-len(tool_results) :]
            )
            round_has_progress = any(
                tc.tool_name == "report_progress"
                for tc in all_tool_calls[-len(tool_results) :]
            )
            if round_has_progress:
                consecutive_exec_python = 0
            elif round_has_exec:
                consecutive_exec_python += 1

            # Progress check: after N consecutive execute_python rounds, require report_progress
            if (
                consecutive_exec_python >= config.progress_check_interval
                and consecutive_exec_python % config.progress_check_interval == 0
            ):
                _progress_msg = (
                    "PROGRESS CHECK: You have run {n} consecutive computations without "
                    "reporting your progress. Before your next execute_python call, "
                    "you MUST call `report_progress` to summarize your findings so far "
                    "and state whether you are ready to call `{exit_tool}`.\n\n"
                    "If your recent computations confirmed what earlier ones already "
                    "showed, you likely have enough evidence — set ready_to_conclude "
                    "to true."
                ).format(n=consecutive_exec_python, exit_tool=_exit_tool_msg)
                messages.append({"role": "user", "content": _progress_msg})
                # Expose report_progress tool for the next round
                if hasattr(tool_executor, "_progress_check_pending"):
                    tool_executor._progress_check_pending = True
                round_log.append(
                    {
                        "kind": "scaffold_injection",
                        "round": round_num,
                        "label": "progress_check",
                        "content": _progress_msg,
                    }
                )
                if config.workspace_dir:
                    log_scaffold_event(
                        config.workspace_dir,
                        iteration,
                        CC.CALL_RELIABILITY,
                        "progress_check",
                        f"consecutive_exec={consecutive_exec_python}, round={round_num}",
                    )

            # Final warning near end of loop
            if round_num == max_rounds - 2 and max_rounds >= 5:
                _final_warn_msg = (
                    "WARNING: You have 2 rounds left before your session ends. "
                    "Wrap up your work and prepare your final output. "
                    f"Call {_exit_tool_msg} now."
                )
                messages.append({"role": "user", "content": _final_warn_msg})
                round_log.append(
                    {
                        "kind": "scaffold_injection",
                        "round": round_num,
                        "label": "final_warning",
                        "content": _final_warn_msg,
                    }
                )

            # CRITICAL penultimate-round instruction
            if round_num == max_rounds - 1 and max_rounds >= 4:
                _crit_msg = (
                    "CRITICAL: This is your LAST round with tool access. "
                    f"You MUST call {_exit_tool_msg} now, or write your final "
                    "output as text in this response."
                )
                messages.append({"role": "user", "content": _crit_msg})
                round_log.append(
                    {
                        "kind": "scaffold_injection",
                        "round": round_num,
                        "label": "critical_warning",
                        "content": _crit_msg,
                    }
                )

            # Progressive log: flush after each round that continues the loop
            _flush_log()

    # --- Post-loop: forced final call(s) with exit tool ---
    _reasons_human = {
        "max_rounds": "You have reached the maximum number of tool-use rounds.",
        "tool_call_failure": "The tool-calling interface is unavailable due to a provider error.",
        "provider_side_400": "The tool-calling interface is unavailable due to a provider-side processing error.",
        "empty_end_turn": "Your previous responses produced no output after repeated attempts.",
        "context_too_long": "The conversation context exceeded the model's maximum length. Summarize your findings so far.",
    }
    reason = _reasons_human.get(loop_exit_reason, _reasons_human["max_rounds"])

    if config.workspace_dir:
        log_scaffold_event(
            config.workspace_dir,
            iteration,
            CC.CALL_RELIABILITY,
            "forced_final_call",
            loop_exit_reason,
        )

    # Always include the exit tool unless a provider-side failure prevents tool use
    _forced_with_exit_tool = loop_exit_reason not in (
        "tool_call_failure",
        "provider_side_400",
    )
    _exit_tool_defs = None
    if _forced_with_exit_tool:
        _exit_tool_defs = [
            t for t in tools if t["function"]["name"] in _exit_tool_names
        ]
        if not _exit_tool_defs:
            _forced_with_exit_tool = False

    # Agent-agnostic forced exit message (delivered as user message, not system prompt mutation).
    # Each agent's process_response() handles format normalization for its own output.
    if _forced_with_exit_tool:
        forced_msg = (
            f"IMPORTANT: {reason}\n\n"
            f"You MUST call {_exit_tool_msg} now with your findings.\n"
        )
    else:
        forced_msg = (
            f"IMPORTANT: {reason}\n\n"
            "You cannot call any more tools. You MUST now write your final output as text.\n"
            "Summarize what you have accomplished and provide your conclusions.\n"
        )
    messages.append({"role": "user", "content": forced_msg})

    # Log the injection in conversation log
    round_log.append(
        {
            "kind": "scaffold_injection",
            "round": round_num + 1,
            "label": "forced_final_message",
            "content": forced_msg,
        }
    )

    # Clear executor state so forced dispatch isn't rejected by the dispatch gate
    if hasattr(tool_executor, "begin_round"):
        tool_executor.begin_round()

    # Retry forced final call if model doesn't call exit tool
    _forced_max_retries = 2 if _forced_with_exit_tool else 0
    final_text = ""
    final_in = final_out = final_reasoning = final_answer = 0
    final_dur = 0.0
    for _forced_attempt in range(_forced_max_retries + 1):
        _forced_call_kwargs = dict(
            model=config.model_id,
            max_tokens=config.max_tokens,
            system=system,  # UNCHANGED system prompt
            messages=provider.prepare_messages(messages),
        )
        if _forced_with_exit_tool:
            _forced_call_kwargs["tools"] = _exit_tool_defs
        # else: no tools parameter — forces text-only response
        try:
            start = time.time()
            final_resp = _call_provider_with_retry(
                provider,
                config,
                workspace_dir=config.workspace_dir,
                iteration=iteration,
                **_forced_call_kwargs,
            )
            dur = time.time() - start
            final_in += final_resp.input_tokens
            final_out += final_resp.output_tokens
            final_reasoning += final_resp.reasoning_tokens
            final_answer += final_resp.answer_tokens
            final_dur += dur
            final_text = final_resp.text.strip()

            # Process exit tool call if returned
            if _forced_with_exit_tool and final_resp.tool_calls:
                for tc_info in final_resp.tool_calls:
                    if tc_info["name"] in _exit_tool_names:
                        tc = tool_executor.execute(tc_info["name"], tc_info["input"])
                        all_tool_calls.append(tc)
                        round_log.append(
                            {
                                "kind": "tool_result",
                                "round": round_num + 1 + _forced_attempt,
                                "tool_name": tc.tool_name,
                                "tool_input": tc.tool_input,
                                "output": tc.output,
                                "is_error": tc.is_error,
                                "duration": tc.duration,
                            }
                        )

            # Exit tool called — done
            if getattr(tool_executor, "stop_after_round", False):
                break

            # Retry if exit tool not called and retries remain
            if _forced_with_exit_tool and _forced_attempt < _forced_max_retries:
                messages.append(
                    provider.format_assistant_message(final_resp.raw_content)
                )
                _retry_msg = (
                    f"You did not call {_exit_tool_msg}. "
                    f"You MUST call {_exit_tool_msg} with your findings to complete your task."
                )
                messages.append({"role": "user", "content": _retry_msg})
                round_log.append(
                    {
                        "kind": "scaffold_injection",
                        "round": round_num + 1 + _forced_attempt,
                        "label": "forced_exit_tool_retry",
                        "content": _retry_msg,
                    }
                )
                if config.workspace_dir:
                    log_scaffold_event(
                        config.workspace_dir,
                        iteration,
                        CC.CALL_RELIABILITY,
                        "forced_exit_tool_retry",
                        f"attempt={_forced_attempt + 1}",
                    )
                continue
            break  # No retries configured or all retries exhausted
        except Exception as exc:
            console.print(
                f"[yellow]Forced final call failed: {type(exc).__name__}: {exc}[/yellow]"
            )
            if config.workspace_dir:
                log_scaffold_event(
                    config.workspace_dir,
                    iteration,
                    CC.CALL_RELIABILITY,
                    "forced_final_call_failed",
                    f"{type(exc).__name__}: {str(exc)[:200]}",
                )
            break

    # Token accumulation AFTER retry loop (correct for both success and failure)
    total_input += final_in
    total_output += final_out
    total_reasoning += final_reasoning
    total_answer += final_answer

    # Exit tool successfully called during forced final
    if _forced_with_exit_tool and getattr(tool_executor, "stop_after_round", False):
        if on_round:
            on_round(
                round_num + 1,
                "executor_stop",
                [],
                total_input,
                total_output,
                final_in,
                final_out,
                round_duration=final_dur,
                round_reasoning=final_reasoning,
                round_answer=final_answer,
            )
        round_log.append(
            {
                "kind": "forced_final_call",
                "round": round_num + 1,
                "reason": "forced_exit_tool",
                "text": final_text,
                "input_tokens": final_in,
                "output_tokens": final_out,
                "reasoning_tokens": final_reasoning,
                "answer_tokens": final_answer,
                "duration": final_dur,
            }
        )
        result = AgentResult(
            text=final_text,
            tool_calls=all_tool_calls,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            rounds=round_num + 1 + _forced_attempt,
            truncated=False,
            duration=time.time() - overall_start,
            stop_reason="executor_stop",
            token_alert_fired=token_alert_fired,
            total_reasoning_tokens=total_reasoning,
            total_answer_tokens=total_answer,
        )
        _write_agent_conversation_log(
            config,
            system,
            user_content,
            agent_name,
            iteration,
            round_log,
            result,
            tools=tools,
            _filepath=_log_filepath,
        )
        return result

    if on_round:
        on_round(
            round_num + 1,
            "forced_partial",
            [],
            total_input,
            total_output,
            final_in,
            final_out,
            round_duration=final_dur,
            round_reasoning=final_reasoning,
            round_answer=final_answer,
        )

    round_log.append(
        {
            "kind": "forced_final_call",
            "round": round_num + 1,
            "reason": loop_exit_reason,
            "text": final_text,
            "input_tokens": final_in,
            "output_tokens": final_out,
            "reasoning_tokens": final_reasoning,
            "answer_tokens": final_answer,
            "duration": final_dur,
        }
    )

    if config.workspace_dir:
        log_llm_call(
            config.workspace_dir,
            agent_name,
            iteration,
            config.model,
            final_in,
            final_out,
            "forced_partial",
            _round_num(final_dur, 2),
            len(system),
            len(user_content),
            len(final_text),
            reasoning_tokens=final_reasoning,
            answer_tokens=final_answer,
            round=round_num + 1,
        )

    # Return result — empty text is an honest failure
    result = AgentResult(
        text=final_text,
        tool_calls=all_tool_calls,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        rounds=round_num + 1 + _forced_attempt,
        truncated=True,
        duration=time.time() - overall_start,
        stop_reason="max_rounds_forced",
        token_alert_fired=token_alert_fired,
        total_reasoning_tokens=total_reasoning,
        total_answer_tokens=total_answer,
    )
    _write_agent_conversation_log(
        config,
        system,
        user_content,
        agent_name,
        iteration,
        round_log,
        result,
        tools=tools,
        _filepath=_log_filepath,
    )
    return result


def _allocate_log_path(config: Config, agent_name: str, iteration: int) -> Path | None:
    """Allocate the next log file path and increment the sequence counter.

    Returns *None* when logging is disabled (no ``logs_dir``).
    """
    if not config.logs_dir:
        return None
    seq = _call_seq.get(iteration, 0) + 1
    _call_seq[iteration] = seq
    agent = agent_name or "unknown"
    return Path(config.logs_dir, f"iter{iteration:03d}_{seq:02d}_{agent}.md")


def _write_preliminary_log(
    log_path: Path | None, system: str, user_content: str
) -> None:
    """Write system prompt + user message to *log_path* before the API call.

    If the call later succeeds, the file is overwritten with the full log
    (including the response).  On failure the preliminary version survives,
    giving the operator enough context to diagnose what was sent.
    """
    if log_path is None:
        return
    content = (
        f'<SYSTEM_PROMPT chars="{len(system)}">\n'
        f"{system}\n"
        f"</SYSTEM_PROMPT>\n\n"
        f'<USER_MESSAGE chars="{len(user_content)}">\n'
        f"{user_content}\n"
        f"</USER_MESSAGE>\n\n"
        f'<LLM_RESPONSE status="pending">\n'
        f"(API call in progress)\n"
        f"</LLM_RESPONSE>\n"
    )
    try:
        log_path.write_text(content)
    except OSError:
        pass


def _write_conversation_log(
    config: Config,
    resp: LLMResponse,
    system: str,
    user_content: str,
    agent_name: str,
    iteration: int,
    log_path: Path | None = None,
) -> str:
    """Write a per-call Markdown file with full prompts and response.

    If *log_path* is provided (pre-allocated), the file is overwritten
    in place.  Otherwise a new path is allocated (legacy callers).

    Returns the absolute path of the log file, or "" if not written.
    """
    if log_path is None:
        log_path = _allocate_log_path(config, agent_name, iteration)
    if log_path is None:
        return ""

    resp_attrs = f'chars="{len(resp.text)}" tokens_in="{resp.input_tokens}" tokens_out="{resp.output_tokens}"'
    if resp.reasoning_tokens and resp.reasoning_tokens > 0:
        resp_attrs += (
            f' reasoning="{resp.reasoning_tokens}" answer="{resp.answer_tokens}"'
        )
    resp_attrs += f' duration="{resp.duration:.1f}s" stop="{resp.stop_reason}"'

    content = f"""<SYSTEM_PROMPT chars="{len(system)}">
{system}
</SYSTEM_PROMPT>

<USER_MESSAGE chars="{len(user_content)}">
{user_content}
</USER_MESSAGE>

<LLM_RESPONSE {resp_attrs}>
{resp.text}
</LLM_RESPONSE>
"""
    try:
        log_path.write_text(content)
    except OSError:
        return ""
    return str(log_path)


def _render_tool_input(name: str, input_data) -> str:
    """Render tool input for conversation log display."""
    if (
        name == "execute_python"
        and isinstance(input_data, dict)
        and "code" in input_data
    ):
        purpose = input_data.get("purpose", "")
        purpose_line = f"**Purpose:** {purpose}\n\n" if purpose else ""
        return f"{purpose_line}~~~python\n{input_data['code']}\n~~~"
    if name == "submit_review" and isinstance(input_data, dict):
        verdict = input_data.get("verdict", "?")
        summary = input_data.get("summary", "?")
        return f"**Verdict: {verdict}** — {summary[:200]}"
    if name == "report_progress" and isinstance(input_data, dict):
        ready = "Yes" if input_data.get("ready_to_conclude") else "No"
        findings = input_data.get("findings_so_far", "?")
        return f"**Ready to conclude:** {ready}\n**Findings:** {findings}"
    try:
        return f"```json\n{json.dumps(input_data, indent=2)}\n```"
    except (TypeError, ValueError):
        return f"```\n{input_data}\n```"


def _render_tools_block(tools: list[dict]) -> str:
    """Render a compact TOOLS block listing tool names and descriptions."""
    if not tools:
        return ""
    lines = ["<TOOLS>"]
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "?")
        desc = func.get("description", "")
        # First line of description only, truncated
        first_line = desc.split("\n")[0].strip()[:120]
        lines.append(f"- {name}: {first_line}")
    lines.append("</TOOLS>\n")
    return "\n".join(lines)


def _render_agent_conversation_log(
    system: str,
    user_content: str,
    round_log: list[dict],
    tools: list[dict] | None = None,
) -> str:
    """Render the full Markdown content for a tool-use agent conversation log."""
    lines: list[str] = []

    # System prompt
    lines.append(f'<SYSTEM_PROMPT chars="{len(system)}">')
    lines.append(system)
    lines.append("</SYSTEM_PROMPT>\n")

    # Tools
    if tools:
        lines.append(_render_tools_block(tools))

    # User content
    lines.append(f'<USER_MESSAGE chars="{len(user_content)}">')
    lines.append(user_content)
    lines.append("</USER_MESSAGE>\n")

    # Render chronological events
    round_open = False
    for entry in round_log:
        kind = entry["kind"]

        if kind == "llm_response":
            if round_open:
                lines.append("</ROUND>\n")
            lines.append(f'<ROUND n="{entry["round"]}">')
            round_open = True
            attrs = f'tokens_in="{entry["input_tokens"]}" tokens_out="{entry["output_tokens"]}"'
            if entry.get("reasoning_tokens", 0) > 0:
                attrs += f' reasoning="{entry["reasoning_tokens"]}" answer="{entry["answer_tokens"]}"'
            attrs += (
                f' duration="{entry["duration"]:.1f}s" stop="{entry["stop_reason"]}"'
            )
            lines.append(f"\n<LLM_RESPONSE {attrs}>")
            text = entry.get("text", "")
            if text and text.strip():
                lines.append(text.strip())
            else:
                lines.append("*(no text output)*")
            lines.append("</LLM_RESPONSE>\n")
            if entry.get("tool_calls"):
                for tc_info in entry["tool_calls"]:
                    tc_name = tc_info.get("name", "unknown")
                    tc_input = tc_info.get("input", {})
                    lines.append(f'<TOOL_CALL name="{tc_name}">')
                    lines.append(_render_tool_input(tc_name, tc_input))
                    lines.append("</TOOL_CALL>\n")

        elif kind == "tool_result":
            status = "error" if entry.get("is_error") else "success"
            dur_attr = (
                f' duration="{entry["duration"]:.1f}s"' if entry.get("duration") else ""
            )
            lines.append(
                f'<TOOL_RESULT name="{entry["tool_name"]}"{dur_attr} status="{status}">'
            )
            lines.append(f"```\n{entry['output']}\n```")
            lines.append("</TOOL_RESULT>\n")

        elif kind == "scaffold_injection":
            if round_open:
                lines.append("</ROUND>\n")
                round_open = False
            lines.append(f'<USER_MESSAGE label="scaffold: {entry["label"]}">')
            lines.append(entry.get("content", ""))
            lines.append("</USER_MESSAGE>\n")

        elif kind == "checkpoint_response":
            attrs = f'tokens_in="{entry["input_tokens"]}" tokens_out="{entry["output_tokens"]}"'
            lines.append(f"\n<CHECKPOINT_RESPONSE {attrs}>")
            text = entry.get("text", "")
            if text and text.strip():
                lines.append(text.strip())
            else:
                lines.append("*(no text output)*")
            lines.append("</CHECKPOINT_RESPONSE>\n")

        elif kind == "forced_final_call":
            if round_open:
                lines.append("</ROUND>\n")
                round_open = False
            reason = entry.get("reason", "unknown")
            attrs = f'reason="{reason}" tokens_in="{entry["input_tokens"]}" tokens_out="{entry["output_tokens"]}"'
            if entry.get("reasoning_tokens", 0) > 0:
                attrs += f' reasoning="{entry["reasoning_tokens"]}" answer="{entry["answer_tokens"]}"'
            if entry.get("duration"):
                attrs += f' duration="{entry["duration"]:.1f}s"'
            lines.append(f"<FORCED_FINAL_CALL {attrs}>")
            text = entry.get("text", "")
            if text and text.strip():
                lines.append(text.strip())
            else:
                lines.append("*(no text output)*")
            lines.append("</FORCED_FINAL_CALL>\n")

    if round_open:
        lines.append("</ROUND>\n")

    return "\n".join(lines) + "\n"


def _write_agent_conversation_log(
    config: Config,
    system: str,
    user_content: str,
    agent_name: str,
    iteration: int,
    round_log: list[dict],
    result: AgentResult,
    tools: list[dict] | None = None,
    _filepath: Path | None = None,
):
    """Write a single comprehensive Markdown log for a tool-use agent invocation."""
    if not config.logs_dir:
        return

    if _filepath is None:
        seq = _call_seq.get(iteration, 0) + 1
        _call_seq[iteration] = seq
        agent = agent_name or "unknown"
        _filepath = Path(config.logs_dir, f"iter{iteration:03d}_{seq:02d}_{agent}.md")

    content = _render_agent_conversation_log(system, user_content, round_log, tools)
    try:
        _filepath.write_text(content)
    except OSError:
        pass
