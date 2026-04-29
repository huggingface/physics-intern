#!/usr/bin/env python3
"""Fill missing CritPt submission JSONs with empty template answers.

After a batch run where some problems failed (timeout, parsing, token limit),
this script creates dummy submission JSONs using each problem's answer_template
so the full 70-problem set can be submitted.

Usage:
    uv run python scripts/fill_missing_critpt.py results/critpt_oneshot/nemotron/20260407_124539
    uv run python scripts/fill_missing_critpt.py results/critpt_oneshot/nemotron/20260407_124539 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROBLEMS_DIR = PROJECT_ROOT / "problems" / "critpt" / "yaml"


def discover_all_problem_ids(problems_dir: Path) -> dict[int, Path]:
    """Return {problem_number: yaml_path} for all 70 challenges."""
    pattern = re.compile(r"Challenge_(\d+)_main\.yaml$")
    problems: dict[int, Path] = {}
    for p in sorted(problems_dir.iterdir()):
        m = pattern.match(p.name)
        if m:
            problems[int(m.group(1))] = p
    return problems


def find_existing_submissions(output_dir: Path) -> set[int]:
    """Return problem numbers that already have valid submission JSONs."""
    existing: set[int] = set()
    pattern = re.compile(r"Challenge_(\d+)_main\.json$")
    for f in output_dir.glob("Challenge_*_main.json"):
        m = pattern.match(f.name)
        if not m:
            continue
        try:
            data = json.loads(f.read_text())
            if data.get("problem_id") and data.get("generated_code"):
                existing.add(int(m.group(1)))
        except (json.JSONDecodeError, KeyError):
            pass
    return existing


def get_template_code(yaml_path: Path) -> str:
    """Extract answer_template from a problem YAML file."""
    data = yaml.safe_load(yaml_path.read_text())
    template = data.get("answer_template", "")
    return template.strip()


def infer_model_from_metadata(output_dir: Path) -> tuple[str, dict]:
    """Read batch_metadata.json to get model string and generation_config."""
    meta_path = output_dir / "batch_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta.get("model", "unknown"), meta.get("generation_config", {})
    # Fallback: try to read from any existing submission JSON
    for f in output_dir.glob("Challenge_*_main.json"):
        try:
            data = json.loads(f.read_text())
            return data.get("model", "unknown"), data.get("generation_config", {})
        except (json.JSONDecodeError, KeyError):
            continue
    return "unknown", {}


def main():
    parser = argparse.ArgumentParser(
        description="Fill missing CritPt submissions with empty template answers.",
    )
    parser.add_argument(
        "output_dir", type=Path, help="Results directory containing submission JSONs"
    )
    parser.add_argument(
        "--problems-dir",
        type=Path,
        default=DEFAULT_PROBLEMS_DIR,
        help="Directory of problem YAMLs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without writing files",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        print(f"Error: {args.output_dir} is not a directory", file=sys.stderr)
        return 1

    all_problems = discover_all_problem_ids(args.problems_dir)
    existing = find_existing_submissions(args.output_dir)
    missing = sorted(set(all_problems.keys()) - existing)

    if not missing:
        print("All 70 submissions already present. Nothing to do.")
        return 0

    model, generation_config = infer_model_from_metadata(args.output_dir)
    generation_config["filled_from_template"] = True

    print(f"Output dir: {args.output_dir}")
    print(f"Model:      {model}")
    print(f"Existing:   {len(existing)}/70")
    print(f"Missing:    {len(missing)} — {', '.join(f'C{n}' for n in missing)}")
    print()

    for n in missing:
        problem_id = f"Challenge_{n}_main"
        yaml_path = all_problems[n]
        template_code = get_template_code(yaml_path)

        submission = {
            "problem_id": problem_id,
            "generated_code": template_code,
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation_config": generation_config,
            "messages": [],
        }

        out_path = args.output_dir / f"{problem_id}.json"
        if args.dry_run:
            print(f"  [dry-run] Would create {out_path.name}")
        else:
            out_path.write_text(json.dumps(submission, indent=2, ensure_ascii=False))
            print(f"  Created {out_path.name}")

    if not args.dry_run:
        # Verify final count
        final_count = len(list(args.output_dir.glob("Challenge_*_main.json")))
        print(f"\nDone. Submission JSONs: {final_count}/70")
    else:
        print(f"\n[dry-run] Would bring total to {len(existing) + len(missing)}/70")

    return 0


if __name__ == "__main__":
    sys.exit(main())
