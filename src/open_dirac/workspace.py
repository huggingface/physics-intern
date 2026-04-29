"""Workspace manager for OpenDirac file I/O and git operations."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .utils.markdown import render_frontmatter


class WorkspaceManager:
    """Manages the workspace directory, file I/O, and git."""

    def __init__(self, config: Config):
        self.root = Path(config.workspace_dir)
        self.computations_dir = self.root / "computations"
        self.derivations_dir = self.root / "derivations"
        self.logs_dir = self.root / "logs"

    def attach(self):
        """Attach to an existing workspace for resume (no git init, no MD creation)."""
        if not self.root.exists():
            raise FileNotFoundError(f"Workspace not found: {self.root}")
        if not (self.root / ".git").exists():
            raise FileNotFoundError(f"Workspace has no .git directory: {self.root}")
        self.computations_dir.mkdir(exist_ok=True)
        self.derivations_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

    def init(self, problem: str):
        """Create workspace, initialize all .md files, git init."""
        self.problem_statement = problem.strip()
        self.root.mkdir(parents=True, exist_ok=True)
        self.computations_dir.mkdir(exist_ok=True)
        self.derivations_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # RESEARCH_STATE.md
        research_meta = {
            "problem_id": "research-session",
            "title": problem[:80],
            "status": "not_started",
            "last_updated": now,
            "iteration": 0,
        }
        research_body = f"""# Problem Statement

{problem.strip()}

# Conventions

(To be populated by the orchestrator as conventions become clear.)

# Working Hypotheses (WH) and Established Results (ER)

Claims use ## ER-NNN (established, verified) or ## WH-NNN (working hypothesis, pending).

# Dead Ends

# Open Questions
"""
        self.write_file(
            "RESEARCH_STATE.md", render_frontmatter(research_meta, research_body)
        )

        # CRITIQUE_LOG.md
        critique_meta = {
            "total_critiques": 0,
            "unresolved_critiques": 0,
            "last_critic_pass": "never",
        }
        self.write_file(
            "CRITIQUE_LOG.md",
            render_frontmatter(
                critique_meta, "# Active Critiques\n\n# Resolved Critiques\n"
            ),
        )

        # EVIDENCE_LOG.md
        evidence_meta = {
            "total_entries": 0,
        }
        self.write_file(
            "EVIDENCE_LOG.md", render_frontmatter(evidence_meta, "# Evidence Log\n")
        )

        # METRICS.md
        self.write_file(
            "METRICS.md",
            render_frontmatter(
                {
                    "total_iterations": 0,
                    "total_llm_calls": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "max_tokens_reached_count": 0,
                    "retries": 0,
                },
                "# Per-Iteration Metrics\n\n# Alerts\n",
            ),
        )

        # Git init
        subprocess.run(
            ["git", "init"], cwd=str(self.root), capture_output=True, check=False
        )
        subprocess.run(
            ["git", "add", "-A"], cwd=str(self.root), capture_output=True, check=False
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial workspace setup"],
            cwd=str(self.root),
            capture_output=True,
            check=False,
        )

    def read_file(self, filename: str) -> str:
        """Read a file from the workspace. Returns empty string if missing."""
        path = self.root / filename
        if path.exists():
            return path.read_text()
        return ""

    def write_file(self, filename: str, content: str):
        """Write content to a workspace file."""
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def append_file(self, filename: str, content: str):
        """Append content to a workspace file."""
        path = self.root / filename
        with open(path, "a") as f:
            f.write(content)

    def read_file_tail(self, filename: str, n_entries: int = 5) -> str:
        """Read the last N ## sections from a file."""
        from .utils.markdown import tail_entries

        text = self.read_file(filename)
        if not text:
            return ""
        return tail_entries(text, n_entries)

    def file_size(self, filename: str) -> int:
        """Return the character count of a file. 0 if missing."""
        path = self.root / filename
        if path.exists():
            return len(path.read_text())
        return 0

    def delete_file(self, filename: str):
        """Delete a file from the workspace if it exists."""
        path = self.root / filename
        if path.exists():
            path.unlink()

    def file_exists(self, filename: str) -> bool:
        """Check if a file exists in the workspace."""
        return (self.root / filename).exists()

    def git_commit(self, message: str):
        """Stage all changes and commit.

        Skips silently if the workspace has no .git directory (e.g. the
        workspace was moved while the run was in progress).
        """
        if not (self.root / ".git").exists():
            return
        subprocess.run(
            ["git", "add", "-A"], cwd=str(self.root), capture_output=True, check=False
        )
        subprocess.run(
            ["git", "commit", "-m", message, "--allow-empty"],
            cwd=str(self.root),
            capture_output=True,
            check=False,
        )


def log_scaffold_event(
    workspace_dir: str | Path,
    iteration: int,
    category: str,
    event: str,
    detail: str = "",
) -> None:
    """Append one scaffold event to EVENT_LOG.jsonl. Never raises."""
    try:
        entry = {
            "kind": "scaffold",
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "iter": iteration,
            "category": category,
            "event": event,
            "detail": detail,
        }
        with open(Path(workspace_dir) / "EVENT_LOG.jsonl", "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def log_llm_call(
    workspace_dir: str | Path,
    agent: str,
    iteration: int,
    model: str,
    input_tokens: int,
    output_tokens: int,
    stop_reason: str,
    duration_s: float,
    system_prompt_chars: int,
    user_content_chars: int,
    response_chars: int,
    reasoning_tokens: int = 0,
    answer_tokens: int = 0,
    round: int = 0,
) -> None:
    """Append one LLM-call event to EVENT_LOG.jsonl. Never raises."""
    try:
        entry = {
            "kind": "llm_call",
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "agent": agent,
            "iter": iteration,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "stop_reason": stop_reason,
            "duration_s": duration_s,
            "system_prompt_chars": system_prompt_chars,
            "user_content_chars": user_content_chars,
            "response_chars": response_chars,
            "reasoning_tokens": reasoning_tokens,
            "answer_tokens": answer_tokens,
            "round": round,
        }
        with open(Path(workspace_dir) / "EVENT_LOG.jsonl", "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
