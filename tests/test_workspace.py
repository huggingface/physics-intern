"""Tests for workspace initialization, file I/O, and git operations."""

import inspect

from open_dirac.config import Config
from open_dirac.workspace import WorkspaceManager


def _make_ws(tmp_path, init=True):
    """Helper: create and optionally init a WorkspaceManager."""
    config = Config(workspace_dir=str(tmp_path / "ws"))
    ws = WorkspaceManager(config)
    if init:
        ws.init("Derive the result.")
    return ws


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


class TestInit:
    def test_creates_research_state_without_warmups(self, tmp_path):
        ws = _make_ws(tmp_path)
        content = ws.read_file("RESEARCH_STATE.md")
        assert "# Problem Statement" in content
        assert "# Conventions" in content
        assert "# Working Hypotheses (WH) and Established Results (ER)" in content
        assert "Warm-Up" not in content
        assert "warm_up" not in content
        assert "WU-" not in content

    def test_signature_has_no_warm_ups_param(self):
        sig = inspect.signature(WorkspaceManager.init)
        assert "warm_ups" not in sig.parameters

    def test_creates_directory_structure(self, tmp_path):
        ws = _make_ws(tmp_path)
        assert ws.root.exists()
        assert ws.computations_dir.exists()
        assert ws.derivations_dir.exists()
        assert ws.logs_dir.exists()

    def test_creates_all_md_files(self, tmp_path):
        ws = _make_ws(tmp_path)
        assert ws.file_exists("RESEARCH_STATE.md")
        assert ws.file_exists("CRITIQUE_LOG.md")
        assert ws.file_exists("EVIDENCE_LOG.md")
        assert ws.file_exists("METRICS.md")

    def test_creates_git_repo(self, tmp_path):
        ws = _make_ws(tmp_path)
        assert (ws.root / ".git").exists()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_write_read_roundtrip(self, tmp_path):
        ws = _make_ws(tmp_path)
        ws.write_file("test.txt", "hello world")
        assert ws.read_file("test.txt") == "hello world"

    def test_read_missing_file_returns_empty(self, tmp_path):
        ws = _make_ws(tmp_path)
        assert ws.read_file("nonexistent.txt") == ""

    def test_append_file(self, tmp_path):
        ws = _make_ws(tmp_path)
        ws.write_file("log.txt", "line1\n")
        ws.append_file("log.txt", "line2\n")
        assert ws.read_file("log.txt") == "line1\nline2\n"

    def test_file_exists(self, tmp_path):
        ws = _make_ws(tmp_path)
        assert not ws.file_exists("nope.txt")
        ws.write_file("nope.txt", "content")
        assert ws.file_exists("nope.txt")

    def test_delete_file(self, tmp_path):
        ws = _make_ws(tmp_path)
        ws.write_file("temp.txt", "delete me")
        assert ws.file_exists("temp.txt")
        ws.delete_file("temp.txt")
        assert not ws.file_exists("temp.txt")

    def test_delete_missing_file_no_error(self, tmp_path):
        ws = _make_ws(tmp_path)
        ws.delete_file("does_not_exist.txt")  # should not raise

    def test_file_size(self, tmp_path):
        ws = _make_ws(tmp_path)
        ws.write_file("sized.txt", "abcde")
        assert ws.file_size("sized.txt") == 5

    def test_file_size_missing_returns_zero(self, tmp_path):
        ws = _make_ws(tmp_path)
        assert ws.file_size("missing.txt") == 0

    def test_write_creates_subdirectories(self, tmp_path):
        ws = _make_ws(tmp_path)
        ws.write_file("sub/dir/file.txt", "nested")
        assert ws.read_file("sub/dir/file.txt") == "nested"


# ---------------------------------------------------------------------------
# read_file_tail
# ---------------------------------------------------------------------------


class TestReadFileTail:
    def test_tail_returns_last_sections(self, tmp_path):
        ws = _make_ws(tmp_path)
        content = "## Section 1\nfoo\n\n## Section 2\nbar\n\n## Section 3\nbaz\n"
        ws.write_file("test.md", content)
        tail = ws.read_file_tail("test.md", n_entries=2)
        assert "Section 2" in tail
        assert "Section 3" in tail

    def test_tail_missing_file_returns_empty(self, tmp_path):
        ws = _make_ws(tmp_path)
        assert ws.read_file_tail("missing.md") == ""


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------


class TestGitCommit:
    def test_commit_after_write(self, tmp_path):
        ws = _make_ws(tmp_path)
        ws.write_file("new.txt", "content")
        ws.git_commit("add new.txt")
        # Verify commit exists in log
        import subprocess

        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=str(ws.root),
            capture_output=True,
            text=True,
        )
        assert "add new.txt" in result.stdout

    def test_commit_no_git_dir_silent(self, tmp_path):
        """git_commit is a no-op when .git is missing."""
        config = Config(workspace_dir=str(tmp_path / "no_git"))
        ws = WorkspaceManager(config)
        ws.root.mkdir(parents=True)
        ws.write_file("test.txt", "data")
        ws.git_commit("should not crash")  # no .git → silent no-op
