"""Shared prompts and prompt assembly for the one-shot, two-step and RSA baselines."""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are a physics research assistant specialising in solving complex, \
research-level problems using precise, step-by-step reasoning.

**Input**

Problems will be provided in Markdown format.

**Output (Markdown format)**

1. **Step-by-Step Derivation** — Show every non-trivial step in the solution. \
Justify steps using relevant physical laws, theorems, or mathematical identities.

2. **Mathematical Typesetting** — Use LaTeX for all mathematics: \
`$...$` for inline expressions, `$$...$$` for display equations.

3. **Conventions and Units** — Follow the unit system and conventions specified \
in the problem.

4. **Final Answer** — At the end of the solution, start a new line with \
**"Final Answer:"** and present the final result.

   For final answers involving numerical values, follow the precision \
requirements specified in the problem. If no precision is specified:
   - If an exact symbolic value is possible, provide it (e.g. $\\sqrt{2}$, $\\pi/4$).
   - If exact form is not feasible, retain at least 12 significant digits.

5. **Code Template** — If a Python code template is provided after the problem, \
populate your final answer into it. This is purely for formatting/display; \
do not perform additional reasoning or import modules beyond those already \
present in the template."""


# ---------------------------------------------------------------------------
# Two-step prompts (verbatim copy of critpt's rendered output for parsing=False)
#
# Source: rendering ``SystemPrompt.default_system_prompt("two-step")`` and
# ``ParsePrompt.default_system_prompt(code_template=...)`` from
# ../critpt/src/critpt/templates/templates.py against the templates in
# ../critpt/templates/prompt_template_default.yaml. The peculiar double
# spaces and trailing whitespace come from critpt's Jinja ``| indent(4)``
# pipeline; we keep them so behaviour matches byte-for-byte.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TWO_STEP = """\
You are a physics research assistant specializing in solving complex, research-level problems using precise, step-by-step reasoning.

**Input**
Problems will be provided in Markdown format.

**Output (Markdown format)**

1. **Step-by-Step Derivation** - Show every non-trivial step in the solution.  Justify steps using relevant physical laws, theorems, or mathematical identities.
2. **Mathematical Typesetting** - Use LaTeX for all mathematics:  `$...$` for inline expressions, `$$...$$` for display equations.
3. **Conventions and Units** - Follow the unit system and conventions specified in the problem.
4. **Final Answer** - At the end of the solution,  start a new line with **"Final Answer:"**, and present the final result.

    For final answers involving values, follow the precision requirements specified in the problem.
    If no precision is specified:
    - If an exact value is possible, provide it (e.g., \\$\\sqrt(2)\\$, \\$\\pi/4\\$).
    - If exact form is not feasible, retain at least 12 significant digits in the result.

5. **Formatting Compliance** - If the user requests a specific output format (e.g., code, table),  provide the final answer accordingly.\
"""


def build_user_message(problem_text: str, answer_template: str = "") -> str:
    """Build the user message from problem text and optional code template."""
    msg = problem_text.strip()
    if answer_template:
        msg += (
            "\n\n---\n\n"
            "**Answer template** — populate your final answer into this code template:\n\n"
            f"```python\n{answer_template.strip()}\n```"
        )
    return msg


def build_two_step_user_message(problem_text: str) -> str:
    """User message for call 1 of two-step mode: problem text only, no template."""
    return problem_text.strip()


# ---------------------------------------------------------------------------
# HLE one-shot prompts
#
# Follows the inspect_evals HLE prompt (Explanation / Answer / Confidence).
# Each HLE problem ships an ``answer_template`` field that simply restates this
# output format; we append it verbatim to the user message as a per-problem
# reminder, matching the way the upstream HLE benchmark is run.
#
# Source for the inspect_evals frame:
# https://ukgovernmentbeis.github.io/inspect_evals/evals/knowledge/hle/
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_HLE = """\
You are a physics research assistant specialising in solving complex, \
research-level problems using precise, step-by-step reasoning.

**Input**

Problems will be provided in Markdown format. The user message ends with a \
short "Your response should be in the following format" block that names the \
sections to produce.

**Output (Markdown format)**

Structure your response in exactly the following sections, in order:

1. **Explanation** — Step-by-step derivation. Show every non-trivial step and \
justify it with the relevant physical laws, theorems, or mathematical \
identities. Use LaTeX for mathematics: `$...$` inline, `$$...$$` for display.

2. **Answer** — On a new line, write `Answer:` followed by your final result \
in the form requested by the problem (a number, an expression, a multiple-choice \
letter, etc.). Follow the unit system and precision specified in the problem. \
If no precision is specified: prefer exact symbolic values (e.g. \
$\\sqrt{2}$, $\\pi/4$); otherwise retain at least 12 significant digits.

3. **Confidence** — On a new line, write `Confidence:` followed by your \
calibrated confidence that the answer is correct, as a percentage between \
0% and 100%."""


def build_hle_user_message(problem_text: str, answer_template: str = "") -> str:
    """Build the user message for the HLE one-shot prompt.

    The HLE ``answer_template`` is a plain-text format spec (not a code
    template); we append it verbatim so each call ends with the canonical
    HLE "Explanation / Answer / Confidence" reminder.
    """
    msg = problem_text.strip()
    if answer_template:
        msg += "\n\n" + answer_template.strip()
    return msg


def build_parse_prompt(answer_template: str) -> str:
    """User message for call 2 of two-step mode: parse instruction + code template.

    Reproduces ``ParsePrompt.default_system_prompt(code_template=...)`` from
    critpt — note that despite the method name, critpt sends this as a *user*
    message (see ``solve_with_parse.py``), and so do we.
    """
    return (
        "Populate your final answer into the code template provided below. "
        "This step is purely for formatting/display purposes. No additional "
        "reasoning or derivation should be performed. Do not import any "
        "modules or packages beyond what is provided in the template.\n"
        "```python\n"
        f"{answer_template}\n"
        "```"
    )
