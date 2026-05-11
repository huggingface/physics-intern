"""LLM-as-judge fallback for answer evaluation.

Used when the problem's ``answer_template`` is a plain-text format spec
(e.g. the HLE "Explanation / Answer / Confidence" template) rather than a
Python ``def answer(...)`` template the symbolic comparator can execute.
Dispatched from :mod:`.formal_eval` so callers see the same
:class:`FormalEvalResult` shape regardless of the path taken.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core.config import Config
from ..core.console import console
from ..llm import call_llm

# Judge model is hardcoded — same registry lookup as any other model key.
JUDGE_MODEL_KEY = "gpt-5.5-high"

JUDGE_SYSTEM_PROMPT = (
    "You are an expert judge evaluating physics problem responses. "
    "Decide only whether the response matches the provided correct answer."
)

# Verbatim inspect_evals HLE judge template (sources: see prompts.py).
# Source: https://ukgovernmentbeis.github.io/inspect_evals/evals/knowledge/hle/
JUDGE_PROMPT_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based \
on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. \
Put the extracted answer as 'None' if there is no exact, final answer to \
extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect \
based on [correct_answer], focusing only on if there are meaningful \
differences between [correct_answer] and the extracted_final_answer. Do not \
comment on any background to the problem, do not attempt to solve the \
problem, do not argue for any answer different than [correct_answer], focus \
only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] \
given above, or is within a small margin of error for numerical problems. \
Answer 'no' otherwise, i.e. if there if there is any inconsistency, \
ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from \
[response]. Put 100 if there is no confidence score available.
"""

_CORRECT_RE = re.compile(r"^\s*correct\s*:\s*(yes|no)\b", re.IGNORECASE | re.MULTILINE)


def _parse_correct(text: str) -> bool | None:
    """Find the ``correct: yes|no`` line in the judge output.

    Returns True/False on match, None when the marker is missing/ambiguous.
    """
    matches = _CORRECT_RE.findall(text or "")
    if not matches:
        return None
    # If multiple ``correct:`` lines appear, trust the last one — the judge
    # sometimes restates the rubric before its final verdict.
    return matches[-1].strip().lower() == "yes"


def run_llm_judge(workspace_dir: str, problem_def: dict):
    """Judge the answer in ``workspace_dir/ANSWER.md`` against the YAML ground truth.

    Returns a :class:`FormalEvalResult` so the caller (``run_formal_evaluation``)
    can treat both paths uniformly. Import is deferred to avoid a circular
    import with :mod:`.formal_eval`.
    """
    from .formal_eval import FormalEvalResult

    answer_path = Path(workspace_dir) / "ANSWER.md"
    if not answer_path.exists():
        return FormalEvalResult(
            skipped=True, skip_reason="ANSWER.md not found in workspace"
        )
    response_text = answer_path.read_text()
    if not response_text.strip():
        return FormalEvalResult(skipped=True, skip_reason="ANSWER.md is empty")

    question = str(problem_def.get("problem", "")).strip()
    correct_answer = str(problem_def.get("answer", "")).strip()
    if not question or not correct_answer:
        return FormalEvalResult(
            skipped=True,
            skip_reason="Missing 'problem' or 'answer' field for LLM judge",
        )

    user_content = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        response=response_text.strip(),
        correct_answer=correct_answer,
    )

    try:
        judge_config = Config(model=JUDGE_MODEL_KEY, workspace_dir=workspace_dir)
    except Exception as exc:
        return FormalEvalResult(
            correct=None,
            method="llm_judge_error",
            error=f"Failed to build judge config ({JUDGE_MODEL_KEY}): {exc}",
        )

    console.print(f"  [dim]LLM judge: calling {JUDGE_MODEL_KEY}...[/]")
    try:
        resp = call_llm(
            JUDGE_SYSTEM_PROMPT,
            user_content,
            judge_config,
            agent_name="judge",
        )
    except Exception as exc:
        return FormalEvalResult(
            correct=None,
            method="llm_judge_error",
            error=f"{type(exc).__name__}: {exc}",
        )

    console.print(
        f"  [dim]LLM judge: {resp.input_tokens} in / {resp.output_tokens} out, "
        f"{resp.duration:.1f}s[/]"
    )

    verdict = _parse_correct(resp.text)
    if verdict is None:
        return FormalEvalResult(
            correct=None,
            method="llm_judge_parse_error",
            error="Could not find 'correct: yes|no' in judge response",
            details=(resp.text or "")[:500],
        )

    return FormalEvalResult(
        correct=verdict,
        method="llm_judge",
        details=f"judge={JUDGE_MODEL_KEY}",
    )
