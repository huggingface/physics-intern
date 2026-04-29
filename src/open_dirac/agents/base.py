"""Base agent with template method pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

from ..config import Config
from ..console import console
from ..llm import (
    AgentResult,
    ContextTooLongError,
    LLMResponse,
    ParseFailureError,
    call_llm,
    call_llm_continuation,
)
from ..metrics import MetricsTracker
from ..tool_call import ToolCall
from ..utils.categories import CompensationCategory as CC
from ..workspace import WorkspaceManager, log_scaffold_event

if TYPE_CHECKING:
    from ..task import Task


class BaseAgent(ABC):
    """Abstract base for all OpenDirac agents."""

    name: str = "base"
    prompt_file: str = ""
    tools: ClassVar[list[dict]] = []
    max_tool_rounds: ClassVar[int | None] = None
    raise_on_parse_failure: ClassVar[bool] = False
    # parse_retries is now read from self.config.parse_retries (config.default.yaml)

    def __init__(
        self, config: Config, workspace: WorkspaceManager, metrics: MetricsTracker
    ):
        self.config = config
        self.workspace = workspace
        self.metrics = metrics
        self._system_prompt: str | None = None
        self._last_script_names: list[str] = []

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            import inspect

            subclass_dir = Path(inspect.getfile(type(self))).parent
            path = subclass_dir / self.prompt_file
            self._system_prompt = path.read_text()
        return self._system_prompt

    @abstractmethod
    def build_context(self, task: Task, iteration: int) -> str:
        """Build the user message context for the LLM call."""
        ...

    @abstractmethod
    def process_response(
        self, response: LLMResponse | AgentResult, task: Task, iteration: int
    ):
        """Process the LLM response: write files, execute code, etc."""
        ...

    def run(
        self,
        task: Task,
        iteration: int,
        on_round: Callable[[int, str, list[ToolCall], int, int, int, int, float], None]
        | None = None,
    ) -> LLMResponse | AgentResult:
        """Template method: build context -> call LLM -> process response."""
        context = self.build_context(task, iteration)
        if self.tools:
            response = self._call_with_tools(
                context, task, iteration, on_round=on_round
            )
        else:
            response = self._call_with_retry(context, iteration)
        self.process_response(response, task, iteration)
        return response

    def _call_with_tools(
        self,
        context: str,
        task: Task,
        iteration: int,
        on_round: Callable[[int, str, list[ToolCall], int, int, int, int, float], None]
        | None = None,
    ) -> AgentResult:
        """Run the tool-use agent loop. Subclasses with ``tools`` must override."""
        raise NotImplementedError(
            f"{type(self).__name__} declares tools but does not implement _call_with_tools"
        )

    def _validate_response(self, response: LLMResponse) -> bool:
        """Check whether the one-shot response is parseable.

        Override in subclasses to enable parse-failure retry. Return False
        to trigger a retry (up to ``parse_retries`` additional attempts).
        """
        return True

    def _parse_retry_hint(self, parse_error: str | None = None) -> str | None:
        """Return a format reminder for the continuation retry, or None.

        Override in subclasses that produce structured JSON output to provide
        the expected JSON schema.  When non-None and ``parse_retries > 0``,
        ``_call_with_retry`` will re-send the original context together with
        the model's previous (incomplete) response and this hint, instead of
        making a blind fresh retry.
        """
        return None

    def _call_with_retry(self, context: str, iteration: int) -> LLMResponse:
        """Call LLM for one-shot agents, with two-phase retry on parse failure.

        Phase 1 — *in-conversation continuation*: sends the LLM's own response
        back as an assistant turn, followed by a user message containing the
        specific JSON parse error and the format hint.  This is cheap (prompt
        caching) and lets the model see exactly what it produced.

        Phase 2 — *fresh call*: if the continuation didn't help, starts a clean
        conversation with the error diagnostic embedded in the context.

        Both phases use the JSON error from :func:`extract_json_with_error` so
        the model knows *what* broke, not just that something was wrong.

        Truncation (``stop_reason == "max_tokens"``) is handled by ``call_llm``
        itself via :func:`continue_on_max_tokens`; this method only retries on
        structural JSON parse failures.
        """
        from .parsing import extract_json_with_error

        max_retries = self.config.parse_retries

        response = call_llm(
            self.system_prompt,
            context,
            self.config,
            agent_name=self.name,
            iteration=iteration,
        )

        self.metrics.record_call(
            iteration=iteration,
            agent=self.name,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            duration=response.duration,
            max_tokens_hit=(response.stop_reason == "max_tokens"),
            reasoning_tokens=response.reasoning_tokens,
            answer_tokens=response.answer_tokens,
        )

        # If call_llm's continuation retries were exhausted and the response
        # is still truncated, surface the alert now — independent of whether
        # a downstream JSON parse retry also runs below.
        if response.stop_reason == "max_tokens":
            self._log_max_tokens(response, iteration)

        # Accumulate across retries for merged response
        accumulated_text = response.text or ""
        total_input = response.input_tokens
        total_output = response.output_tokens
        total_duration = response.duration
        total_reasoning = response.reasoning_tokens
        total_answer = response.answer_tokens

        attempted_retry = False
        for attempt in range(max_retries):
            # Decide whether a retry is needed. Note: truncation is already
            # retried upstream in call_llm via continue_on_max_tokens; here
            # we only retry on structural JSON parse failures.
            needs_retry = not self._validate_response(response)

            if not needs_retry:
                return response  # valid — done

            if not accumulated_text.strip():
                break  # empty text — can't continue

            # Get specific parse error for the feedback message
            _, parse_error = extract_json_with_error(accumulated_text)

            hint = self._parse_retry_hint(parse_error=parse_error)
            if not hint:
                break  # no hint available — can't continue

            attempted_retry = True

            # Truncate error for console display
            error_short = (parse_error or "unknown")[:120]
            phase = "in-conversation" if attempt == 0 else "fresh"
            console.print(
                f"[yellow]{self.name}: parse error — {error_short}[/yellow]\n"
                f"[yellow]          retrying {phase} ({attempt + 1}/{max_retries})...[/yellow]"
            )
            log_scaffold_event(
                self.workspace.root,
                iteration,
                category=CC.OUTPUT_NORMALIZATION,
                event="parse_retry",
                detail=(
                    f"agent={self.name}, trigger={response.stop_reason}, "
                    f"phase={phase}, attempt={attempt + 1}/{max_retries}, "
                    f"error={error_short}"
                ),
            )

            # Build the correction message
            error_line = (
                f"Your JSON output failed to parse: {parse_error}\n\n"
                if parse_error
                else "Your response did not contain valid structured JSON.\n\n"
            )
            correction = (
                error_line + "Do not repeat the analysis. Output ONLY the corrected "
                "fenced ```json``` block.\n\n" + hint
            )

            try:
                if attempt == 0:
                    # Phase 1: multi-turn continuation
                    messages = [
                        {"role": "user", "content": context},
                        {"role": "assistant", "content": accumulated_text},
                        {"role": "user", "content": correction},
                    ]
                    retry = call_llm_continuation(
                        self.system_prompt,
                        messages,
                        self.config,
                        agent_name=self.name,
                        iteration=iteration,
                        append_to_log=response.log_path,
                    )
                else:
                    # Phase 2: fresh call with error context
                    retry_context = f"{context}\n\n---\n\n" + correction
                    retry = call_llm(
                        self.system_prompt,
                        retry_context,
                        self.config,
                        agent_name=self.name,
                        iteration=iteration,
                    )
            except ContextTooLongError:
                console.print(
                    f"[yellow]{self.name}: context too long during parse "
                    f"retry (attempt {attempt + 1}) — returning best "
                    f"response so far[/yellow]"
                )
                log_scaffold_event(
                    self.workspace.root,
                    iteration,
                    category=CC.OUTPUT_NORMALIZATION,
                    event="parse_retry_context_too_long",
                    detail=f"agent={self.name}, attempt={attempt + 1}",
                )
                return response

            self.metrics.record_call(
                iteration=iteration,
                agent=self.name,
                input_tokens=retry.input_tokens,
                output_tokens=retry.output_tokens,
                duration=retry.duration,
                max_tokens_hit=(retry.stop_reason == "max_tokens"),
                reasoning_tokens=retry.reasoning_tokens,
                answer_tokens=retry.answer_tokens,
            )

            # Accumulate
            accumulated_text = accumulated_text + "\n\n" + (retry.text or "")
            total_input += retry.input_tokens
            total_output += retry.output_tokens
            total_duration += retry.duration
            total_reasoning += retry.reasoning_tokens
            total_answer += retry.answer_tokens

            # Build merged response for validation
            response = LLMResponse(
                text=accumulated_text,
                input_tokens=total_input,
                output_tokens=total_output,
                stop_reason=retry.stop_reason,
                duration=total_duration,
                reasoning_tokens=total_reasoning,
                answer_tokens=total_answer,
            )

            if self._validate_response(response):
                console.print(
                    f"[green]{self.name}: parse retry succeeded "
                    f"(attempt {attempt + 1})[/green]"
                )
                log_scaffold_event(
                    self.workspace.root,
                    iteration,
                    category=CC.OUTPUT_NORMALIZATION,
                    event="parse_retry_success",
                    detail=f"agent={self.name}, phase={phase}, attempt={attempt + 1}",
                )
                return response

        # All retries exhausted or no hint available
        if attempted_retry:
            console.print(
                f"[red]{self.name}: parse retries exhausted "
                f"({max_retries}/{max_retries})[/red]"
            )
            log_scaffold_event(
                self.workspace.root,
                iteration,
                category=CC.OUTPUT_NORMALIZATION,
                event="parse_retry_failed",
                detail=f"agent={self.name}",
            )

        # Raise if agent opts in and response is still invalid
        if self.raise_on_parse_failure and not self._validate_response(response):
            raise ParseFailureError(
                agent_name=self.name,
                detail=f"stop_reason={response.stop_reason}, "
                f"text_len={len(response.text or '')}",
            )

        return response

    def _log_max_tokens(self, response: LLMResponse, iteration: int) -> None:
        """Log max_tokens alert when call_llm's continuation retries are exhausted."""
        self.metrics.alert(
            iteration,
            f"max_tokens_reached on {self.name} "
            f"(input={response.input_tokens}, output={response.output_tokens})",
        )
        log_scaffold_event(
            self.workspace.root,
            iteration,
            category=CC.LOOP_CONTROL,
            event="max_tokens_unrecoverable",
            detail=f"agent={self.name}, output_tokens={response.output_tokens}",
        )
