"""Tests for the resume feature: config persistence, workspace attach, engine resume."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from open_dirac.config import Config, _PERSIST_FIELDS
from open_dirac.engine import (
    OpenDirac,
    LoopState,
    _reconstruct_loop_state,
    _find_last_critic_iteration,
)
from open_dirac.workspace import WorkspaceManager
from open_dirac.research_state import (
    ResearchState,
    Hypothesis,
    HypothesisStatus,
    Evidence,
    ReviewResult,
    Verdict,
    ResearchQuestion,
)


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


class TestConfigPersistence:
    def test_config_save_load_roundtrip(self, tmp_path):
        config = Config(workspace_dir=str(tmp_path))
        config.save(tmp_path)
        loaded = Config.load(tmp_path)
        for f in _PERSIST_FIELDS:
            assert getattr(loaded, f) == getattr(config, f), f"Mismatch on {f}"

    def test_config_load_with_overrides(self, tmp_path):
        config = Config(max_iterations=5, workspace_dir=str(tmp_path))
        config.save(tmp_path)
        loaded = Config.load(tmp_path, overrides={"max_iterations": 20})
        assert loaded.max_iterations == 20

    def test_config_excludes_sensitive_fields(self, tmp_path):
        config = Config(workspace_dir=str(tmp_path), api_key="secret-key-123")
        config.save(tmp_path)
        data = json.loads((tmp_path / "config.json").read_text())
        assert "api_key" not in data
        assert "logs_dir" not in data
        assert "workspace_dir" not in data

    def test_config_load_with_model_override_re_resolves(self, tmp_path):
        config = Config(workspace_dir=str(tmp_path))
        config.save(tmp_path)
        # Overriding model should clear provider/model_id so __post_init__
        # re-resolves them from models.yaml (along with max_tokens).
        loaded = Config.load(tmp_path, overrides={"model": "claude-4.6-opus"})
        assert loaded.model == "claude-4.6-opus"
        assert loaded.provider == "anthropic"
        assert loaded.model_id == "claude-opus-4-6"
        assert loaded.max_tokens == 128000  # re-resolved from models.yaml

    def test_config_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No config.json"):
            Config.load(tmp_path)

    def test_config_to_dict_fields(self):
        config = Config()
        d = config.to_dict()
        assert set(d.keys()) == _PERSIST_FIELDS

    def test_config_load_none_overrides_ignored(self, tmp_path):
        config = Config(max_iterations=5, workspace_dir=str(tmp_path))
        config.save(tmp_path)
        loaded = Config.load(tmp_path, overrides={"max_iterations": None})
        assert loaded.max_iterations == 5


# ---------------------------------------------------------------------------
# Workspace attach
# ---------------------------------------------------------------------------


class TestWorkspaceAttach:
    def test_workspace_attach_existing(self, tmp_path):
        # Set up a valid workspace
        config = Config(workspace_dir=str(tmp_path))
        ws = WorkspaceManager(config)
        ws.init("test problem")
        # Now attach to it
        ws2 = WorkspaceManager(config)
        ws2.attach()
        assert ws2.computations_dir.exists()
        assert ws2.logs_dir.exists()

    def test_workspace_attach_missing(self, tmp_path):
        config = Config(workspace_dir=str(tmp_path / "nonexistent"))
        ws = WorkspaceManager(config)
        with pytest.raises(FileNotFoundError, match="Workspace not found"):
            ws.attach()

    def test_workspace_attach_no_git(self, tmp_path):
        # Directory exists but no .git
        ws_dir = tmp_path / "no_git_ws"
        ws_dir.mkdir()
        config = Config(workspace_dir=str(ws_dir))
        ws = WorkspaceManager(config)
        with pytest.raises(FileNotFoundError, match="no .git directory"):
            ws.attach()


# ---------------------------------------------------------------------------
# Loop state reconstruction
# ---------------------------------------------------------------------------


class TestReconstructLoopState:
    def test_reconstruct_loop_state_claim_failures(self):
        state = ResearchState()
        # REFUTED review on a WORKING hypothesis → counted
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="A",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict=Verdict.REFUTED, summary="bad", iteration=3),
            iteration_created=1,
            iteration_modified=3,
        )
        # VERIFIED review on a WORKING hypothesis → not counted
        state.hypotheses["WH-002"] = Hypothesis(
            id="WH-002",
            statement="B",
            status=HypothesisStatus.WORKING,
            review=ReviewResult(verdict=Verdict.VERIFIED, summary="ok", iteration=4),
            iteration_created=2,
            iteration_modified=4,
        )
        ls = _reconstruct_loop_state(state)
        assert ls.claim_failure_count == {"WH-001": 1}

    def test_reconstruct_loop_state_last_content_iter(self):
        state = ResearchState()
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="A",
            status=HypothesisStatus.WORKING,
            evidence=[Evidence(type="research", result="x", iteration=5)],
            review=ReviewResult(verdict=Verdict.VERIFIED, summary="ok", iteration=7),
            iteration_created=1,
            iteration_modified=7,
        )
        state.research_questions["RQ-001"] = ResearchQuestion(
            id="RQ-001",
            question="Q?",
            evidence=[Evidence(type="compute", result="y", iteration=6)],
            iteration_created=2,
        )
        ls = _reconstruct_loop_state(state)
        assert ls.last_content_iteration == 7  # max of 5, 7, 6

    def test_reconstruct_loop_state_pending_empty(self):
        state = ResearchState()
        ls = _reconstruct_loop_state(state)
        assert ls.pending_violations == []
        assert ls.pending_termination_blockers == []
        assert ls.pending_compute_verdicts == []
        assert ls.pending_verified_results == []
        assert ls.pending_explore_results == []
        assert ls.agent_failures == []

    def test_reconstruct_loop_state_established_not_counted(self):
        """REFUTED review on an ESTABLISHED hypothesis should NOT be counted."""
        state = ResearchState()
        state.hypotheses["ER-001"] = Hypothesis(
            id="ER-001",
            statement="A",
            status=HypothesisStatus.ESTABLISHED,
            review=ReviewResult(verdict=Verdict.REFUTED, summary="bad", iteration=3),
            iteration_created=1,
            iteration_modified=3,
        )
        ls = _reconstruct_loop_state(state)
        assert ls.claim_failure_count == {}


# ---------------------------------------------------------------------------
# Find last critic iteration
# ---------------------------------------------------------------------------


class TestFindLastCriticIteration:
    def test_find_last_critic_iteration(self, tmp_path):
        log = tmp_path / "EVENT_LOG.jsonl"
        lines = [
            json.dumps({"kind": "llm_call", "agent": "orchestrator", "iter": 1}),
            json.dumps({"kind": "llm_call", "agent": "deep_critic", "iter": 3}),
            json.dumps({"kind": "llm_call", "agent": "deep_critic", "iter": 6}),
            json.dumps({"kind": "scaffold", "event": "forced_critic", "iter": 6}),
            json.dumps({"kind": "llm_call", "agent": "researcher", "iter": 7}),
        ]
        log.write_text("\n".join(lines) + "\n")
        assert _find_last_critic_iteration(tmp_path) == 6

    def test_find_last_critic_no_file(self, tmp_path):
        assert _find_last_critic_iteration(tmp_path) == 0

    def test_find_last_critic_no_critic_entries(self, tmp_path):
        log = tmp_path / "EVENT_LOG.jsonl"
        log.write_text(
            json.dumps({"kind": "llm_call", "agent": "researcher", "iter": 5}) + "\n"
        )
        assert _find_last_critic_iteration(tmp_path) == 0


# ---------------------------------------------------------------------------
# Engine resume
# ---------------------------------------------------------------------------


class TestEngineResume:
    def _make_workspace(self, tmp_path):
        """Create a minimal valid workspace for resume testing."""
        import subprocess
        import yaml as _yaml

        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()
        (ws_dir / "computations").mkdir()
        (ws_dir / "archive").mkdir()
        (ws_dir / "logs").mkdir()

        # Git init
        subprocess.run(
            ["git", "init"], cwd=str(ws_dir), capture_output=True, check=True
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(ws_dir),
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(ws_dir),
            capture_output=True,
            check=True,
        )

        # problem.yaml
        problem_def = {
            "problem": "Derive the Hawking temperature.",
            "answer_template": "T_H = ...",
        }
        (ws_dir / "problem.yaml").write_text(_yaml.dump(problem_def))

        # config.json
        config = Config(workspace_dir=str(ws_dir))
        config.save(ws_dir)

        # Research state
        state = ResearchState(
            iteration=4,
            status="in_progress",
            title="test-run",
            problem_statement="Derive the Hawking temperature.",
            survey_background="Survey done.",
        )
        state.hypotheses["WH-001"] = Hypothesis(
            id="WH-001",
            statement="T_H = 1/(8*pi*M)",
            status=HypothesisStatus.WORKING,
            evidence=[Evidence(type="research", result="derived", iteration=2)],
            iteration_created=1,
            iteration_modified=2,
        )
        state.save(ws_dir)

        # Initial commit
        subprocess.run(
            ["git", "add", "-A"], cwd=str(ws_dir), capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "state"],
            cwd=str(ws_dir),
            capture_output=True,
            check=True,
        )

        return ws_dir

    def test_resume_loads_research_state(self, tmp_path):
        ws_dir = self._make_workspace(tmp_path)
        engine = OpenDirac.resume(ws_dir)
        assert engine.research_state.status == "in_progress"
        assert "WH-001" in engine.research_state.hypotheses

    def test_resume_sets_iteration(self, tmp_path):
        ws_dir = self._make_workspace(tmp_path)
        engine = OpenDirac.resume(ws_dir)
        assert engine.iteration == 4

    def test_resume_missing_problem_yaml(self, tmp_path):
        import subprocess

        ws_dir = tmp_path / "ws"
        ws_dir.mkdir()
        subprocess.run(
            ["git", "init"], cwd=str(ws_dir), capture_output=True, check=True
        )
        config = Config(workspace_dir=str(ws_dir))
        config.save(ws_dir)
        with pytest.raises(FileNotFoundError, match="problem.yaml"):
            OpenDirac.resume(ws_dir)

    def test_resume_with_config_overrides(self, tmp_path):
        ws_dir = self._make_workspace(tmp_path)
        engine = OpenDirac.resume(ws_dir, config_overrides={"max_iterations": 50})
        assert engine.config.max_iterations == 50

    def test_resume_reconstructs_loop_state(self, tmp_path):
        ws_dir = self._make_workspace(tmp_path)
        engine = OpenDirac.resume(ws_dir)
        # WH-001 has no review, so claim_failure_count should be empty
        assert engine._state.claim_failure_count == {}
        # Evidence at iteration 2
        assert engine._state.last_content_iteration == 2


# ---------------------------------------------------------------------------
# Surveyor skip and completed workspace guard
# ---------------------------------------------------------------------------


class TestRunSurveyorSkip:
    def test_run_skips_surveyor_on_resume(self, tmp_path):
        """With existing survey_background, surveyor should not be called."""
        engine = OpenDirac.__new__(OpenDirac)
        engine.config = Config(workspace_dir=str(tmp_path), max_iterations=0)
        engine.workspace = MagicMock()
        engine.workspace.root = tmp_path
        engine.metrics = MagicMock()
        engine.metrics.calls = []
        engine.metrics.alerts = []
        engine.metrics.total_input_tokens = 0
        engine.metrics.total_output_tokens = 0
        engine.metrics.last_critic_iteration = 0
        engine.iteration = 0
        engine._state = LoopState()
        engine.research_state = ResearchState(
            problem_statement="test",
            survey_background="Already surveyed.",
        )
        engine.problem_meta = {}
        engine.surveyor = MagicMock()
        engine.planner = MagicMock()
        engine.planner.parsed_strategy = None

        engine.run()

        engine.surveyor.run.assert_not_called()

    def test_run_calls_surveyor_fresh(self, tmp_path):
        """Without background survey, surveyor should be called."""
        engine = OpenDirac.__new__(OpenDirac)
        engine.config = Config(workspace_dir=str(tmp_path), max_iterations=0)
        engine.workspace = MagicMock()
        engine.workspace.root = tmp_path
        engine.metrics = MagicMock()
        engine.metrics.calls = []
        engine.metrics.alerts = []
        engine.metrics.total_input_tokens = 0
        engine.metrics.total_output_tokens = 0
        engine.metrics.last_critic_iteration = 0
        engine.iteration = 0
        engine._state = LoopState()
        engine.research_state = ResearchState(problem_statement="test")
        engine.problem_meta = {}
        engine.surveyor = MagicMock()
        engine.surveyor.parsed_survey = None
        engine.planner = MagicMock()
        engine.planner.parsed_strategy = None

        engine.run()

        engine.surveyor.run.assert_called_once()

    def test_completed_workspace_exits_early(self, tmp_path):
        """Status 'completed' should exit without running the loop."""
        engine = OpenDirac.__new__(OpenDirac)
        engine.config = Config(workspace_dir=str(tmp_path), max_iterations=10)
        engine.workspace = MagicMock()
        engine.workspace.root = tmp_path
        engine.metrics = MagicMock()
        engine.metrics.calls = []
        engine.metrics.alerts = []
        engine.metrics.total_input_tokens = 0
        engine.metrics.total_output_tokens = 0
        engine.metrics.last_critic_iteration = 0
        engine.iteration = 5
        engine._state = LoopState()
        engine.research_state = ResearchState(
            problem_statement="test",
            status="completed",
            survey_background="Done.",
        )
        engine.problem_meta = {}
        engine.surveyor = MagicMock()
        engine.planner = MagicMock()
        engine.planner.parsed_strategy = None
        engine.orchestrator = MagicMock()

        engine.run()

        # Orchestrator should never have been called
        engine.orchestrator.run.assert_not_called()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    def test_cli_resume_flag(self):
        from open_dirac.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["--resume", "/some/workspace"])
        assert args.resume == Path("/some/workspace")
        # problem is optional with --resume, defaults to None
        assert args.problem is None

    def test_cli_problem_optional_with_resume(self):
        from open_dirac.main import build_parser

        parser = build_parser()
        # Should not error when problem is omitted with --resume
        args = parser.parse_args(
            ["--resume", "/some/workspace", "--max-iterations", "20"]
        )
        assert args.resume == Path("/some/workspace")
        assert args.max_iterations == 20
        # problem is optional with --resume, defaults to None
        assert args.problem is None

    def test_cli_problem_required_without_resume(self):
        from open_dirac.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["some/problem.yaml"])
        assert args.problem == Path("some/problem.yaml")
        assert args.resume is None
