"""Single LLM call helper shared by the one-shot and RSA baselines.

Wraps ``providers.retry.call_with_retry`` with an agent-agnostic retry
callback that logs transient errors to stderr (baselines have no workspace
log channel by default), applies ``llm.continue_on_max_tokens`` when the
response is truncated, and computes USD cost from the model-registry
entries on the ``Config``. Returns a structured dict.
"""

from __future__ import annotations

import sys
import time

from ..config import Config
from ..llm import continue_on_max_tokens
from ..providers import LLMProvider, call_with_retry


def run_baseline_call(
    provider: LLMProvider,
    config: Config,
    *,
    system: str,
    user_message: str,
    agent_name: str,
) -> dict:
    """Execute one LLM call with retry + max-tokens continuation + cost calc.

    Returns a dict with keys::

        tokens: {"input", "output", "reasoning", "answer"}
        duration_s, cost_usd, stop_reason, response_text
    """

    def _on_retry(exc: Exception, attempt: int, max_retries: int) -> None:
        print(
            f"  Transient error (attempt {attempt + 1}/{max_retries}): {exc}",
            file=sys.stderr,
        )

    start = time.time()
    initial_messages = [{"role": "user", "content": user_message}]
    resp = call_with_retry(
        provider,
        max_retries=config.api_retry_max,
        initial_delay=config.api_retry_initial_delay,
        max_delay=config.api_retry_max_delay,
        on_retry=_on_retry,
        model=config.model_id,
        max_tokens=config.max_tokens,
        system=system,
        messages=initial_messages,
    )
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
            agent_name=agent_name,
        )
    duration = time.time() - start

    cost_usd = 0.0
    if config.input_cost or config.output_cost:
        cost_usd = (
            resp.input_tokens * config.input_cost
            + resp.output_tokens * config.output_cost
        ) / 1_000_000

    return {
        "tokens": {
            "input": resp.input_tokens,
            "output": resp.output_tokens,
            "reasoning": resp.reasoning_tokens or 0,
            "answer": resp.answer_tokens or 0,
        },
        "duration_s": round(duration, 2),
        "cost_usd": round(cost_usd, 6),
        "stop_reason": resp.stop_reason,
        "response_text": resp.text,
    }
