"""CLI entry point for the diagnosis pass on a completed workspace.

The verification package is layered:

- :mod:`.evaluate`      — pure symbolic/numerical comparison (no I/O)
- :mod:`.workspace`     — workspace loading and reference-file lookup
- :mod:`.formal_eval`   — formal answer evaluation (consumed by the engine)
- :mod:`.event_summary` — EVENT_LOG.jsonl → diagnosis-prompt text blocks
- :mod:`.diagnosis`     — LLM-driven audit of a completed run
- :mod:`.cli`           — CLI glue (this file)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import yaml

from ..config import Config, DEFAULTS
from ..console import console
from .diagnosis import (
    build_diagnosis_prompt,
    call_diagnosis_llm,
    parse_diagnosis,
    render_diagnosis,
    write_diagnosis_report,
)
from .formal_eval import load_or_run_formal_eval, render_formal_evaluation
from .workspace import load_reference_file, load_workspace


def build_verify_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for verification."""
    parser = argparse.ArgumentParser(
        prog="open_dirac.verification",
        description="Diagnosis of OpenDirac research workspaces.",
    )
    parser.add_argument("workspace_dir", type=Path, help="Path to workspace directory")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULTS["verify_model"],
        help=f"LLM model (default: {DEFAULTS['verify_model']})",
    )
    return parser


def main():
    parser = build_verify_parser()
    args = parser.parse_args()

    workspace_dir = args.workspace_dir
    if not Path(workspace_dir).is_dir():
        print(f"Error: workspace directory not found: {workspace_dir}")
        sys.exit(1)

    model = args.model

    console.print(f"[bold]Loading workspace:[/] {workspace_dir}")
    contents = load_workspace(workspace_dir)

    if not contents.research_state:
        console.print("[red]Error: RESEARCH_STATE.md not found or empty.[/]")
        sys.exit(1)

    if contents.terminated_cleanly:
        console.print("[green]Run terminated cleanly.[/]")
    else:
        console.print("[yellow]Warning: run did NOT terminate cleanly.[/]")

    problem_def = None
    known_answer = None
    problem_path = Path(workspace_dir) / "problem.yaml"
    if problem_path.exists():
        with open(problem_path) as f:
            problem_def = yaml.safe_load(f)
        answer_val = problem_def.get("answer")
        if answer_val is not None:
            known_answer = str(answer_val)
            console.print(f"[bold]Known answer:[/] {known_answer}")

    problem_name = problem_def.get("name") if problem_def else None
    ref_lookup_path = Path(problem_name + ".yaml") if problem_name else None

    ref_answer_expr, reference_content = load_reference_file(ref_lookup_path)
    if reference_content:
        console.print("[bold]Reference file:[/] loaded")
    if not known_answer and ref_answer_expr:
        known_answer = ref_answer_expr
        console.print(
            f"[bold]Known answer (from reference):[/] {known_answer[:100]}..."
        )

    console.print("\n[bold]Formal answer evaluation...[/]")
    formal_eval = load_or_run_formal_eval(workspace_dir, problem_def, ref_lookup_path)
    render_formal_evaluation(formal_eval)

    config = Config(model=model, workspace_dir=workspace_dir)
    system, user_content = build_diagnosis_prompt(
        contents,
        formal_eval=formal_eval,
        known_answer=known_answer,
        reference_content=reference_content,
    )

    console.print(f"\n[bold]Diagnosis ({model}, streaming)...[/]")
    response = call_diagnosis_llm(system, user_content, config)
    console.print(
        f"[dim]({response.input_tokens} in / {response.output_tokens} out, {response.duration:.1f}s)[/]"
    )

    result = parse_diagnosis(response.text, formal_eval=formal_eval)
    render_diagnosis(result)
    write_diagnosis_report(result, workspace_dir)


if __name__ == "__main__":
    main()
