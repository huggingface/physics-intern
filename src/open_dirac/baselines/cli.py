"""CLI scaffolding shared by the one-shot and RSA baselines.

Centralises argparse registration, problem YAML loading + validation,
workspace directory setup, and provider instantiation. Each baseline's
``main()`` stays in its own ``runner.py`` but reuses these helpers.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

from ..config import DEFAULTS, Config
from ..providers import LLMProvider, create_provider


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register ``problem``, ``--model``, ``--config``, ``-o/--output``,
    and ``--workspace-dir`` on *parser*.

    Baseline-specific flags (e.g. RSA's ``-N -K -T``) are added by each
    runner after calling this helper.
    """
    parser.add_argument("problem", type=Path, help="Path to problem YAML file")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model key from models.yaml (default: {DEFAULTS['model']})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML file (overrides defaults)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save response text to a Markdown file",
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default=None,
        help="Workspace directory (default: auto-generated)",
    )


def load_problem(problem_path: Path) -> tuple[dict, str, str]:
    """Load a problem YAML file.

    Returns ``(problem_def, problem_text, answer_template)``. Exits with
    code 1 on missing file or a missing ``problem`` field.
    """
    if not problem_path.exists():
        print(f"Error: problem file not found: {problem_path}", file=sys.stderr)
        sys.exit(1)
    with open(problem_path) as f:
        problem_def = yaml.safe_load(f)
    problem_text = problem_def.get("problem", "")
    answer_template = problem_def.get("answer_template", "")
    if not problem_text:
        print("Error: problem YAML has no 'problem' field", file=sys.stderr)
        sys.exit(1)
    return problem_def, problem_text, answer_template


def setup_workspace(
    args: argparse.Namespace,
    config: Config,
    problem_def: dict,
    problem_text: str,
    suffix: str,
) -> Path:
    """Create the workspace dir and seed it with PROBLEM.md + problem.yaml + config.json.

    *suffix* is the baseline tag (e.g. ``"oneshot"``, ``"rsa"``) used in the
    auto-generated directory name. Mutates ``config.workspace_dir`` in place
    so downstream calls that rely on it (e.g. ``continue_on_max_tokens``
    scaffold logging) pick up the location.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = config.model.replace("/", "-").replace(":", "-")
    workspace_root = Path(
        args.workspace_dir
        or f"workspaces/{timestamp}_{args.problem.stem}_{safe_model}_{suffix}"
    )
    workspace_root.mkdir(parents=True, exist_ok=True)
    config.workspace_dir = str(workspace_root)

    (workspace_root / "PROBLEM.md").write_text(f"# Problem\n\n{problem_text}\n")
    problem_data = dict(problem_def)
    problem_data["name"] = args.problem.stem
    with open(workspace_root / "problem.yaml", "w") as f:
        yaml.dump(problem_data, f, default_flow_style=False, sort_keys=False)
    config.save(workspace_root)
    return workspace_root


def create_provider_from_config(config: Config) -> LLMProvider:
    """Instantiate the provider both baselines use the same way."""
    return create_provider(
        config.provider,
        api_key=config.api_key,
        timeout=config.api_timeout,
        **config.reasoning,
    )
