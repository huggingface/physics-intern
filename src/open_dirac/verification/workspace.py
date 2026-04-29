"""Workspace loading and reference-file lookup for the verification subsystem."""

from __future__ import annotations

import glob
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.markdown import parse_frontmatter
from ..utils.sandbox import ExecutionResult, execute_python

# Resolved relative to this file's location: <repo>/src/open_dirac/verification/workspace.py
# → <repo>/references. Fragile if the package moves; revisit via Config if that happens.
REFERENCES_DIR = Path(__file__).parent.parent.parent.parent / "references"

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class WorkspaceContents:
    """Loaded workspace files and metadata."""

    workspace_dir: str
    research_state: str = ""
    critique_log: str = ""
    evidence_log: str = ""
    current_task: str = ""
    computation_scripts: list[str] = field(default_factory=list)
    terminated_cleanly: bool = False
    frontmatter: dict = field(default_factory=dict)
    metrics_md: str = ""
    git_log: str = ""
    event_log: str = ""
    background_survey: str = ""


@dataclass
class RerunResult:
    """Result of re-running a computation script."""

    script_path: str
    execution: ExecutionResult | None = None


def load_workspace(workspace_dir: str) -> WorkspaceContents:
    """Read all relevant files from a completed workspace."""
    ws = Path(workspace_dir)
    contents = WorkspaceContents(workspace_dir=workspace_dir)

    for fname, attr in [
        ("RESEARCH_STATE.md", "research_state"),
        ("CRITIQUE_LOG.md", "critique_log"),
        ("EVIDENCE_LOG.md", "evidence_log"),
        ("CURRENT_TASK.md", "current_task"),
    ]:
        path = ws / fname
        if path.exists():
            setattr(contents, attr, path.read_text())

    # Parse frontmatter from CURRENT_TASK to check termination
    if contents.current_task:
        fm, _ = parse_frontmatter(contents.current_task)
        contents.frontmatter = fm
        contents.terminated_cleanly = fm.get("task_type") == "terminate"

    # Also check RESEARCH_STATE status — the engine may exit via
    # _should_terminate() which sets status: completed/partially_complete
    # without writing a terminate task to CURRENT_TASK.
    if not contents.terminated_cleanly and contents.research_state:
        rs_fm, _ = parse_frontmatter(contents.research_state)
        rs_status = rs_fm.get("status", "")
        if rs_status in ("completed", "partially_complete"):
            contents.terminated_cleanly = True

    scripts = sorted(glob.glob(str(ws / "computations" / "*.py")))
    contents.computation_scripts = scripts

    metrics_path = ws / "METRICS.md"
    if metrics_path.exists():
        contents.metrics_md = metrics_path.read_text()

    event_log_path = ws / "EVENT_LOG.jsonl"
    if event_log_path.exists():
        contents.event_log = event_log_path.read_text()

    # Background survey from JSON state (not in markdown snapshots)
    from ..research_state import STATE_FILENAME
    from ..rendering import render_background_survey

    state_path = ws / STATE_FILENAME
    if state_path.exists():
        try:
            from ..research_state import ResearchState

            state = ResearchState.from_json(state_path.read_text())
            if state.survey_background:
                contents.background_survey = render_background_survey(state)
        except Exception:
            pass  # Non-critical — proceed without survey

    try:
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            contents.git_log = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # Not a git repo or git not available

    return contents


def load_reference_file(problem_path: Path | None) -> tuple[str | None, str | None]:
    """Load reference file matching a problem YAML, if it exists.

    Returns:
        (answer_expr, full_content) where answer_expr is the content
        of the first code block (a SymPy assignment string) and
        full_content is the entire file text.  Both are None if no
        reference file is found.
    """
    if problem_path is None:
        return None, None

    ref_path = REFERENCES_DIR / f"{problem_path.stem}.md"
    if not ref_path.exists():
        return None, None

    content = ref_path.read_text()
    m = _CODE_FENCE_RE.search(content)
    answer_expr = m.group(1).strip() if m else None
    return answer_expr, content


def rerun_computations(workspace_dir: str, timeout: int = 60) -> list[RerunResult]:
    """Re-run all computation scripts in the workspace."""
    ws = Path(workspace_dir)
    scripts = sorted(glob.glob(str(ws / "computations" / "*.py")))
    results = []
    for script in scripts:
        execution = execute_python(script, timeout=timeout, cwd=workspace_dir)
        results.append(RerunResult(script_path=script, execution=execution))
    return results
