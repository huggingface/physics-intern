"""Sub-agent dispatch with optional code execution and retry."""

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..llm import call_llm, call_llm_continuation
from ..utils.sandbox import execute_python


CODE_EXECUTION_SUFFIX = """

## Code execution instructions

You will write Python code to perform the requested computation. Follow these rules:

1. Write your reasoning and explanation as normal text.
2. Write exactly ONE Python code block using triple backticks with the `python` language tag.
3. The script must be completely self-contained: include all imports and definitions.
4. Print all results to stdout. The printed output is what will be returned.
5. Available packages: Python 3.12+, NumPy, SciPy, SymPy, matplotlib, standard library.
6. Do NOT call plt.show() — use plt.savefig() then plt.close().
7. Timeout: 60 seconds. Keep computations efficient.
"""


@dataclass
class SubAgentResult:
    """Result from a sub-agent dispatch."""

    reasoning_text: str
    code: str
    execution_output: str
    execution_status: str  # "success", "failed_after_retries", "no_code"
    total_input_tokens: int
    total_output_tokens: int

    def format_for_manager(self) -> str:
        """Format as a string to return to the Manager."""
        parts = [f"<subagent_reasoning>\n{self.reasoning_text}\n</subagent_reasoning>"]
        if self.code:
            parts.append(f"\n\n<code>\n{self.code}\n</code>")
            parts.append(
                f'\n\n<execution_output status="{self.execution_status}">\n'
                f"{self.execution_output}\n</execution_output>"
            )
        return "\n".join(parts)


def _extract_python_code(text: str) -> str:
    """Extract the first Python code block from text."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _truncate(text: str, limit: int = 10_000) -> str:
    """Truncate output, preserving head and tail."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n\n[... truncated {len(text) - limit} chars ...]\n\n"
        + text[-half:]
    )


def dispatch_subagent(
    system_prompt: str,
    user_message: str,
    execute_code: bool,
    config: Config,
    workspace_root: Path,
    iteration: int,
    subagent_counter: int = 0,
    sandbox_timeout: int = 60,
    max_retries: int = 3,
) -> SubAgentResult:
    """Dispatch an ephemeral sub-agent LLM call.

    If *execute_code* is True, extracts Python code from the response,
    executes in a sandbox, and retries up to *max_retries* times on failure.
    """
    total_in = 0
    total_out = 0

    effective_system = system_prompt
    if execute_code:
        effective_system = system_prompt + CODE_EXECUTION_SUFFIX

    agent_label = f"subagent_iter{iteration}_{subagent_counter}"

    # Initial LLM call
    resp = call_llm(
        system=effective_system,
        user_content=user_message,
        config=config,
        agent_name=agent_label,
        iteration=iteration,
    )
    total_in += resp.input_tokens
    total_out += resp.output_tokens

    if not execute_code:
        return SubAgentResult(
            reasoning_text=resp.text,
            code="",
            execution_output="",
            execution_status="no_code",
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )

    # --- Code execution path ---
    code = _extract_python_code(resp.text)
    # Strip the code block from reasoning to avoid duplication in format_for_manager
    if code:
        reasoning_text = re.sub(
            r"```(?:python)?\s*\n.*?```",
            "",
            resp.text,
            count=1,
            flags=re.DOTALL,
        ).strip()
    else:
        reasoning_text = resp.text

    if not code:
        return SubAgentResult(
            reasoning_text=reasoning_text,
            code="",
            execution_output="No Python code block found in sub-agent response.",
            execution_status="no_code",
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )

    computations_dir = workspace_root / "computations"
    computations_dir.mkdir(parents=True, exist_ok=True)

    last_error = ""
    log_path = resp.log_path

    for attempt in range(1, max_retries + 1):
        script_name = f"{agent_label}_attempt{attempt}.py"
        script_path = computations_dir / script_name
        script_path.write_text(code)

        result = execute_python(
            script_path,
            timeout=sandbox_timeout,
            cwd=str(computations_dir),
        )

        if result.returncode == 0 and not result.timed_out:
            output = result.stdout
            if result.stderr:
                output += f"\n\nSTDERR (warnings):\n{result.stderr}"
            return SubAgentResult(
                reasoning_text=reasoning_text,
                code=code,
                execution_output=_truncate(output),
                execution_status="success",
                total_input_tokens=total_in,
                total_output_tokens=total_out,
            )

        # Build error description
        if result.timed_out:
            last_error = f"TIMEOUT: Script exceeded {sandbox_timeout}s limit."
        else:
            last_error = result.stderr or f"Exit code {result.returncode}"
            if result.stdout:
                last_error = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{last_error}"
        last_error = _truncate(last_error, 5_000)

        if attempt < max_retries:
            retry_msg = (
                f"Your code failed (attempt {attempt}/{max_retries}).\n\n"
                f"Error:\n```\n{last_error}\n```\n\n"
                f"Please fix the code and try again. Write the complete corrected "
                f"script in a single Python code block."
            )
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content": retry_msg},
            ]
            retry_resp = call_llm_continuation(
                system=effective_system,
                messages=messages,
                config=config,
                agent_name=f"{agent_label}_retry{attempt}",
                iteration=iteration,
                append_to_log=log_path,
            )
            total_in += retry_resp.input_tokens
            total_out += retry_resp.output_tokens
            resp = retry_resp
            reasoning_text = resp.text
            new_code = _extract_python_code(resp.text)
            if new_code:
                code = new_code

    # All retries exhausted
    return SubAgentResult(
        reasoning_text=reasoning_text,
        code=code,
        execution_output=f"{last_error}\n\nExecution failed after {max_retries} attempts.",
        execution_status="failed_after_retries",
        total_input_tokens=total_in,
        total_output_tokens=total_out,
    )
