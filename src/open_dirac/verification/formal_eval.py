"""Formal (symbolic/numerical) answer evaluation against ground truth.

This is the stable API consumed by the engine, RSA, and Autophysicist runners
at the end of a research run. It wraps :mod:`open_dirac.verification.evaluate`
with the workspace/problem-def conventions the scaffolding uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..console import console
from ..utils.markdown import parse_frontmatter
from .evaluate import evaluate_response
from .workspace import load_reference_file


@dataclass
class FormalEvalResult:
    """Result of formal (symbolic/numerical) answer evaluation."""

    correct: bool | None = None  # True/False/None (errored)
    method: str = ""
    error: str | None = None
    details: str = ""
    skipped: bool = False
    skip_reason: str = ""


def run_formal_evaluation(
    workspace_dir: str,
    problem_def: dict | None,
    *,
    problem_path: Path | None = None,
) -> FormalEvalResult:
    """Run formal (symbolic/numerical) answer evaluation against ground truth.

    Preconditions (all must be met, otherwise skip gracefully):
    1. problem_def is provided and has an 'answer' field (or reference file fallback)
    2. problem_def has an 'answer_template' field
    3. ANSWER.md exists in the workspace
    """
    if not problem_def:
        return FormalEvalResult(
            skipped=True, skip_reason="No problem definition or no answer field"
        )

    answer_val = problem_def.get("answer")
    answer_empty = answer_val is None or (
        isinstance(answer_val, str) and not answer_val.strip()
    )

    reference_code: str | None = None
    if answer_empty:
        ref_answer, _ = load_reference_file(problem_path)
        if ref_answer:
            if "def answer" in ref_answer:
                reference_code = ref_answer
                console.print("  [dim]Using answer function from reference file[/]")
            else:
                problem_def = dict(
                    problem_def
                )  # shallow copy to avoid mutating caller's dict
                problem_def["answer"] = ref_answer
                console.print("  [dim]Using answer from reference file[/]")
        else:
            return FormalEvalResult(
                skipped=True, skip_reason="No problem definition or no answer field"
            )

    if not problem_def.get("answer_template"):
        return FormalEvalResult(
            skipped=True, skip_reason="No answer_template in problem definition"
        )

    answer_path = Path(workspace_dir) / "ANSWER.md"
    if not answer_path.exists():
        return FormalEvalResult(
            skipped=True, skip_reason="ANSWER.md not found in workspace"
        )

    raw_content = answer_path.read_text()
    if not raw_content.strip():
        return FormalEvalResult(skipped=True, skip_reason="ANSWER.md is empty")

    if "```python" in raw_content:
        fenced_content = raw_content
    else:
        fenced_content = f"```python\n{raw_content}\n```"

    try:
        eval_result = evaluate_response(
            fenced_content,
            problem_def,
            reference_code=reference_code,
        )
    except Exception as exc:
        return FormalEvalResult(
            correct=None,
            method="evaluation_error",
            error=str(exc),
            details="Exception during evaluate_response()",
        )

    return FormalEvalResult(
        correct=eval_result.get("correct"),
        method=eval_result.get("method", ""),
        error=eval_result.get("error"),
        details=eval_result.get("details", ""),
    )


def render_formal_evaluation(result: FormalEvalResult) -> None:
    """Print the formal evaluation result to the console using Rich."""
    if result.skipped:
        console.print(f"[dim]  Skipped: {result.skip_reason}[/]")
        return

    if result.correct is True:
        console.print("[green bold]  CORRECT[/]")
    elif result.correct is False:
        console.print("[red bold]  INCORRECT[/]")
    else:
        console.print("[yellow bold]  INCONCLUSIVE[/]")

    if result.method:
        console.print(f"  Method: {result.method}")
    if result.error:
        console.print(f"  [red]Error: {result.error}[/]")
    if result.details:
        console.print(f"  [dim]{result.details}[/]")
    console.print()


def write_formal_eval_report(result: FormalEvalResult, workspace_dir: str) -> None:
    """Write VERIFICATION.md with just the formal evaluation section.

    Called by the engine at end of run. The diagnosis pass later appends to this file.
    """
    lines = ["---"]
    if result.skipped:
        lines.append("formal_answer: skipped")
    elif result.correct is True:
        lines.append("formal_answer: correct")
    elif result.correct is False:
        lines.append("formal_answer: incorrect")
    else:
        lines.append("formal_answer: inconclusive")
    lines.append("---\n")

    lines.append("# Verification Report\n")

    if result.skipped:
        lines.append("## Formal Answer Evaluation: SKIPPED\n")
        lines.append(f"- Reason: {result.skip_reason}\n")
    else:
        if result.correct is True:
            verdict_str = "CORRECT"
        elif result.correct is False:
            verdict_str = "INCORRECT"
        else:
            verdict_str = "INCONCLUSIVE"
        lines.append(f"## Formal Answer Evaluation: {verdict_str}\n")
        if result.method:
            lines.append(f"- Method: {result.method}")
        if result.error:
            lines.append(f"- Error: {result.error}")
        if result.details:
            lines.append(f"- Details: {result.details}")
        lines.append("")

    report_path = Path(workspace_dir) / "VERIFICATION.md"
    report_path.write_text("\n".join(lines))
    console.print(f"[green]Formal evaluation written to {report_path}[/]")


def load_or_run_formal_eval(
    workspace_dir: str,
    problem_def: dict | None,
    ref_lookup_path: Path | None,
) -> FormalEvalResult:
    """Read formal eval from existing VERIFICATION.md, or run it fresh."""
    report_path = Path(workspace_dir) / "VERIFICATION.md"
    if report_path.exists():
        content = report_path.read_text()
        fm, _ = parse_frontmatter(content)
        formal_answer = fm.get("formal_answer", "")
        if formal_answer == "correct":
            return FormalEvalResult(correct=True, method="from_report")
        elif formal_answer == "incorrect":
            return FormalEvalResult(correct=False, method="from_report")
        elif formal_answer == "inconclusive":
            return FormalEvalResult(correct=None, method="from_report")
        elif formal_answer == "skipped":
            return FormalEvalResult(skipped=True, skip_reason="from_report")

    return run_formal_evaluation(
        workspace_dir, problem_def, problem_path=ref_lookup_path
    )
