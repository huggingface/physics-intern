"""Shared primitives for the one-shot, two-step and RSA baseline runners."""

from .call import run_baseline_call, run_two_step_call
from .cli import (
    add_common_args,
    create_provider_from_config,
    load_problem,
    setup_workspace,
)
from .prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_HLE,
    SYSTEM_PROMPT_TWO_STEP,
    build_hle_user_message,
    build_parse_prompt,
    build_two_step_user_message,
    build_user_message,
)

__all__ = [
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_HLE",
    "SYSTEM_PROMPT_TWO_STEP",
    "build_user_message",
    "build_hle_user_message",
    "build_two_step_user_message",
    "build_parse_prompt",
    "run_baseline_call",
    "run_two_step_call",
    "add_common_args",
    "load_problem",
    "setup_workspace",
    "create_provider_from_config",
]
