#!/usr/bin/env python3
"""One-shot LLM baseline for OpenDirac problems.

Sends a single LLM call with the problem statement and collects the response.
Uses the provider layer (open_dirac.providers + open_dirac.config) but nothing from
the engine, agents, or tools — suitable for comparing raw model capability
against the multi-agent scaffolding.

Usage:
    uv run python -m open_dirac.one_shot problems/critpt/quantum_error_correction_main.yaml
    uv run python -m open_dirac.one_shot problems/critpt/quantum_error_correction_main.yaml --model gpt-5.4-high
    uv run python -m open_dirac.one_shot problems/critpt/quantum_error_correction_main.yaml -o result.md
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from ..baselines import (
    SYSTEM_PROMPT,
    add_common_args,
    build_user_message,
    create_provider_from_config,
    load_problem,
    run_baseline_call,
    setup_workspace,
)
from ..config import Config, build_config
from ..providers import LLMProvider
from ..verification import (
    extract_answer_code,
    run_formal_evaluation,
    write_formal_eval_report,
)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


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
        system=SYSTEM_PROMPT,
        user_message=user_message,
        agent_name="one_shot",
    )

    # --- Report stats to stderr ---
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

    # --- Persist answer to workspace ---
    # Extract only the code block with `def answer` to keep ANSWER.md clean;
    # fall back to the full response if no fenced block is found.
    clean_code = extract_answer_code(result["response_text"])
    answer_content = clean_code if clean_code else result["response_text"]
    (workspace_root / "ANSWER.md").write_text(answer_content + "\n")

    # --- Formal evaluation (writes VERIFICATION.md with frontmatter) ---
    # We deliberately do not call render_formal_evaluation: it writes to the
    # Rich console singleton (stdout), which would corrupt one-shot's
    # stdout-is-response contract. The stderr verdict line below replaces it.
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

    # --- Output response to stdout ---
    print(result["response_text"])

    # --- Optional -o: persist just the response text ---
    if args.output:
        args.output.write_text(result["response_text"])
        print(f"Saved to {args.output}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="open_dirac.one_shot",
        description="One-shot LLM baseline for OpenDirac problems.",
    )
    add_common_args(parser)
    args = parser.parse_args()

    # --- Load problem YAML ---
    problem_def, problem_text, answer_template = load_problem(args.problem)

    # --- Model / provider resolution ---
    config = build_config(args)
    provider = create_provider_from_config(config)

    # --- Build prompt ---
    user_message = build_user_message(problem_text, answer_template)

    # --- Workspace (lightweight, no git) ---
    workspace_root = setup_workspace(
        args,
        config,
        problem_def,
        problem_text,
        "oneshot",
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
