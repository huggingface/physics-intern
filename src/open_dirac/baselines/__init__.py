"""Shared primitives for the one-shot and RSA baseline runners."""

from .call import run_baseline_call
from .cli import (
    add_common_args,
    create_provider_from_config,
    load_problem,
    setup_workspace,
)
from .prompts import SYSTEM_PROMPT, build_user_message

__all__ = [
    "SYSTEM_PROMPT",
    "build_user_message",
    "run_baseline_call",
    "add_common_args",
    "load_problem",
    "setup_workspace",
    "create_provider_from_config",
]
