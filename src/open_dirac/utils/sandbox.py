"""Sandboxed Python script execution."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    """Result of a sandboxed Python execution."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


def execute_python(
    script_path: str | Path, timeout: int = 60, cwd: str | Path | None = None
) -> ExecutionResult:
    """Execute a Python script with timeout. Returns ExecutionResult."""
    script_path = Path(script_path).resolve()
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            stdout="",
            stderr=f"TIMEOUT: Script exceeded {timeout}s limit.",
            returncode=-1,
            timed_out=True,
        )
    except Exception as e:
        return ExecutionResult(
            stdout="",
            stderr=f"EXECUTION ERROR: {e}",
            returncode=-1,
            timed_out=False,
        )
