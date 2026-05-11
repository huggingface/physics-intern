#!/usr/bin/env python3
"""One-shot LLM baseline for HLE-physics problems.

Sends a single LLM call with the HLE-style system prompt (Explanation /
Answer / Confidence + populated answer_template) and collects the response.

Mirrors :mod:`physics_intern.one_shot.runner` but kept independent so the HLE
prompt can evolve without touching the CritPt baselines.

Usage:
    uv run python -m physics_intern.hle_one_shot problems/hle-physics/66b727d3_antisymmetrized_gamma_matrices_dimensions.yaml
    uv run python -m physics_intern.hle_one_shot <path> --model gpt-5.4-high
    uv run python -m physics_intern.hle_one_shot <path> -o result.md
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from ..baselines import (
    SYSTEM_PROMPT_HLE,
    add_common_args,
    build_hle_user_message,
    create_provider_from_config,
    load_problem,
    run_baseline_call,
    setup_workspace,
)
from ..core.config import Config, build_config
from ..providers import LLMProvider
from ..verification import (
    run_formal_evaluation,
    write_formal_eval_report,
)


def _run_single(
    args: argparse.Namespace,
    config: Config,
    provider: LLMProvider,
    user_message: str,
    problem_def: dict,
    workspace_root,
) -> None:
    """Run once, evaluate against ground truth, write outputs."""
    result = run_baseline_call(
        provider,
        config,
        system=SYSTEM_PROMPT_HLE,
        user_message=user_message,
        agent_name="hle_one_shot",
    )

    tokens = result["tokens"]
    print(f"Input tokens:  {tokens['input']}", file=sys.stderr)
    print(f"Output tokens: {tokens['output']}", file=sys.stderr)
    if tokens["reasoning"]:
        print(f"  Reasoning:   {tokens['reasoning']}", file=sys.stderr)
        print(f"  Answer:      {tokens['answer']}", file=sys.stderr)
    print(f"Duration:      {result['duration_s']:.1f}s", file=sys.stderr)
    print(f"Stop reason:   {result['stop_reason']}", file=sys.stderr)
    if result["cost_usd"]:
        print(f"Est. cost:     ${result['cost_usd']:.4f}", file=sys.stderr)

    # HLE responses are plain Explanation / Answer / Confidence text — there
    # is no python code block to extract. Save the response verbatim.
    response_text = result["response_text"] or ""
    (workspace_root / "ANSWER.md").write_text(response_text.rstrip() + "\n")

    ev = run_formal_evaluation(
        str(workspace_root),
        problem_def,
        problem_path=args.problem,
    )
    if ev.skipped:
        print(f"Evaluation:    SKIPPED ({ev.skip_reason})", file=sys.stderr)
    elif ev.correct is True:
        print(f"Evaluation:    CORRECT ({ev.method})", file=sys.stderr)
    elif ev.correct is False:
        print(f"Evaluation:    INCORRECT ({ev.method})", file=sys.stderr)
    else:
        print(f"Evaluation:    ERROR — {ev.error}", file=sys.stderr)
    write_formal_eval_report(ev, str(workspace_root))

    print("---", file=sys.stderr)

    # stdout = full response (the batch runner consumes this verbatim).
    print(result["response_text"])

    if args.output:
        args.output.write_text(result["response_text"])
        print(f"Saved to {args.output}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="physics_intern.hle_one_shot",
        description="One-shot LLM baseline for HLE-physics problems.",
    )
    add_common_args(parser)
    args = parser.parse_args()

    problem_def, problem_text, answer_template = load_problem(args.problem)

    config = build_config(args)
    provider = create_provider_from_config(config)

    user_message = build_hle_user_message(problem_text, answer_template)

    workspace_root = setup_workspace(
        args,
        config,
        problem_def,
        problem_text,
        "hle_oneshot",
    )

    print(f"Model:     {config.model} ({config.model_id})", file=sys.stderr)
    print(f"Provider:  {config.provider}", file=sys.stderr)
    print(f"Problem:   {args.problem.name}", file=sys.stderr)
    print(f"Tokens:    {config.max_tokens} max output", file=sys.stderr)
    print(f"Workspace: {workspace_root}", file=sys.stderr)
    print("---", file=sys.stderr)

    _run_single(args, config, provider, user_message, problem_def, workspace_root)


if __name__ == "__main__":
    main()
