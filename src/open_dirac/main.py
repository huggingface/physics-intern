"""OpenDirac entry point."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import yaml

from .config import build_config
from .engine import OpenDirac


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="open_dirac",
        description="Multi-agent scaffolding for autonomous scientific research.",
    )
    parser.add_argument(
        "problem", type=Path, nargs="?", default=None, help="Path to problem YAML file"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to workspace directory to resume",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Replay console log from a workspace (no run)",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to config YAML file"
    )
    parser.add_argument("--model", type=str, default=None, help="LLM model to use")
    parser.add_argument(
        "--max-iterations", type=int, default=None, help="Maximum iterations"
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help="Workspace directory (default: auto-generated)",
    )
    return parser


def _handle_dirty_workspace(workspace_path: Path) -> None:
    """Check for uncommitted changes and prompt user to clean them."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(workspace_path),
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = result.stdout.strip()
    if not dirty:
        return

    print(f"Warning: workspace has uncommitted changes:\n{dirty}")
    answer = input("Reset to last committed state? [y/N] ").strip().lower()
    if answer == "y":
        subprocess.run(
            ["git", "checkout", "."],
            cwd=str(workspace_path),
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=str(workspace_path),
            capture_output=True,
            check=False,
        )
        print("Workspace cleaned.")
    else:
        print("Aborting resume — clean the workspace first.")
        sys.exit(1)


def _main_fresh(args) -> None:
    """Run a fresh research session."""
    if not args.problem.exists():
        print(f"Error: problem file not found: {args.problem}")
        sys.exit(1)

    with open(args.problem) as f:
        problem_def = yaml.safe_load(f)

    problem = problem_def.get("problem", "")
    answer_template = problem_def.get("answer_template", "")

    # Build config (3-tier merge)
    config = build_config(args)

    # Generate timestamped workspace directory if not explicitly set
    if args.workspace_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = config.model.replace("/", "-").replace(":", "-")
        run_name = f"{timestamp}_{args.problem.stem}_{safe_model}_open_dirac"
        config.workspace_dir = str(Path("workspaces") / run_name)  # Config stores str

    # Print config summary
    print(f"Model:    {config.model} ({config.model_id})", file=sys.stderr)
    print(f"Provider: {config.provider}", file=sys.stderr)
    print(f"Problem:  {args.problem.name}", file=sys.stderr)
    print(f"Tokens:   {config.max_tokens} max output", file=sys.stderr)
    print("---", file=sys.stderr)

    # Build problem metadata for termination gates
    problem_meta = {
        "steps": problem_def.get("steps", []),
    }

    # Run
    engine = OpenDirac(
        problem,
        config=config,
        problem_meta=problem_meta,
        answer_template=answer_template,
    )

    # Persist problem.yaml (with name field) and config.json for future resume
    problem_raw = Path(args.problem).read_text()
    problem_data = yaml.safe_load(problem_raw)
    problem_data["name"] = Path(args.problem).stem
    with open(engine.workspace.root / "problem.yaml", "w") as f:
        yaml.dump(problem_data, f, default_flow_style=False, sort_keys=False)
    config.save(engine.workspace.root)
    engine.workspace.git_commit("Persist problem.yaml and config.json for resume")

    engine.run()


def _main_replay(args) -> None:
    """Replay console log from a workspace directory."""
    from .console import replay_log

    workspace_path = args.replay.resolve()
    log_path = workspace_path / "logs" / "console.log"
    if not log_path.exists():
        print(f"Error: no console log at {log_path}")
        sys.exit(1)
    replay_log(log_path, tail=None)  # show entire log


def _main_resume(args) -> None:
    """Resume an interrupted research session."""
    workspace_path = args.resume.resolve()
    if not workspace_path.exists():
        print(f"Error: workspace not found: {workspace_path}")
        sys.exit(1)

    _handle_dirty_workspace(workspace_path)

    # Collect CLI overrides that should apply on resume
    overrides = {}
    for key in ("model", "max_iterations"):
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            overrides[key] = value

    engine = OpenDirac.resume(workspace_path, config_overrides=overrides or None)
    engine.run()


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.replay is not None:
        _main_replay(args)
    elif args.resume is not None:
        _main_resume(args)
    elif args.problem is None:
        parser.error("the following argument is required: problem")
    else:
        _main_fresh(args)


if __name__ == "__main__":
    main()
