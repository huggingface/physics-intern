"""Configuration for OpenDirac."""

import json
import warnings
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
import os

import yaml


# ---------------------------------------------------------------------------
# Package defaults — single source of truth is config.default.yaml
# ---------------------------------------------------------------------------


def _load_package_defaults() -> dict:
    """Load defaults from the config.default.yaml shipped with the package."""
    path = Path(__file__).parent / "config.default.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or not isinstance(data, dict):
        raise RuntimeError(f"Failed to load package defaults from {path}")
    return data


DEFAULTS = _load_package_defaults()


@dataclass
class Config:
    """OpenDirac configuration.

    ``max_tokens`` is the resolved maximum output-token budget per LLM call.
    It is *not* user-configurable — the single source of truth is the
    ``max_output_tokens`` field of the model's entry in ``models.yaml``.
    ``__post_init__`` populates it from there.
    """

    model: str = DEFAULTS["model"]
    verify_model: str = DEFAULTS["verify_model"]
    max_tokens: int = 0  # resolved from models.yaml in __post_init__
    max_iterations: int = DEFAULTS["max_iterations"]
    critic_every_n: int = DEFAULTS["critic_every_n"]
    sympy_timeout_seconds: int = DEFAULTS["sympy_timeout_seconds"]
    max_tool_rounds: int = DEFAULTS["max_tool_rounds"]
    progress_check_interval: int = DEFAULTS["progress_check_interval"]
    computation_token_alert: int = DEFAULTS["computation_token_alert"]
    tool_output_limit: int = DEFAULTS["tool_output_limit"]
    stall_threshold: int = DEFAULTS["stall_threshold"]
    stall_recompute_limit: int = DEFAULTS["stall_recompute_limit"]
    max_termination_retries: int = DEFAULTS["max_termination_retries"]
    min_er_for_completion: int = DEFAULTS["min_er_for_completion"]
    api_retry_max: int = DEFAULTS["api_retry_max"]
    api_retry_initial_delay: float = DEFAULTS["api_retry_initial_delay"]
    api_retry_max_delay: float = DEFAULTS["api_retry_max_delay"]
    api_timeout: float = DEFAULTS["api_timeout"]
    budget_synthesis_margin: int = DEFAULTS["budget_synthesis_margin"]
    orchestrator_comp_log_tail: int = DEFAULTS["orchestrator_comp_log_tail"]
    prior_failure_excerpt_chars: int = DEFAULTS["prior_failure_excerpt_chars"]
    max_open_rqs: int = DEFAULTS["max_open_rqs"]
    rq_evidence_cap: int = DEFAULTS["rq_evidence_cap"]
    max_refuted_retries: int = DEFAULTS["max_refuted_retries"]
    auto_expire_iterations: int = DEFAULTS["auto_expire_iterations"]
    parse_retries: int = DEFAULTS["parse_retries"]
    max_tokens_retries: int = DEFAULTS["max_tokens_retries"]
    pipeline_retry_max: int = DEFAULTS["pipeline_retry_max"]
    provider: str = ""
    workspace_dir: str = ""
    logs_dir: str = ""
    api_key: str = ""
    model_id: str = ""  # Resolved API model ID (from models.yaml)
    input_cost: float = 0.0  # USD per million input tokens (from models.yaml)
    output_cost: float = 0.0  # USD per million output tokens (from models.yaml)
    reasoning: dict = field(default_factory=dict)  # provider-specific reasoning params

    def to_dict(self) -> dict:
        """Serialize config fields for persistence (excludes sensitive/derived fields)."""
        return {f: getattr(self, f) for f in _PERSIST_FIELDS}

    def save(self, workspace_root: Path) -> None:
        """Write config.json to the workspace root."""
        (workspace_root / "config.json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        )

    @classmethod
    def load(cls, workspace_root: Path, overrides: dict | None = None) -> "Config":
        """Load config from workspace config.json, merging optional overrides."""
        path = workspace_root / "config.json"
        if not path.exists():
            raise FileNotFoundError(f"No config.json found in {workspace_root}")
        data = json.loads(path.read_text())
        if overrides:
            # If user switches model, clear provider/model_id so __post_init__ re-resolves
            if "model" in overrides and overrides["model"] is not None:
                data.pop("provider", None)
                data.pop("model_id", None)
                data.pop("input_cost", None)
                data.pop("output_cost", None)
                data.pop("reasoning", None)
            for k, v in overrides.items():
                if v is not None:
                    data[k] = v
        return cls(**data)

    def __post_init__(self):
        resolved = _resolve_model(self.model)
        # Resolve provider from models.yaml if not explicitly set
        if not self.provider:
            if resolved:
                self.provider = resolved["provider"]
                self.model_id = resolved["model_id"]
                self.input_cost = resolved.get("input_cost", 0.0)
                self.output_cost = resolved.get("output_cost", 0.0)
                self.reasoning = resolved.get("reasoning", {})
            else:
                # Default to anthropic for backward compatibility
                self.provider = "anthropic"
        # Resolve max_tokens from models.yaml — the single source of truth.
        # Runs on every init (including resume) so the value always reflects
        # the current registry even if the persisted config.json is stale.
        if resolved:
            model_max = resolved.get("max_output_tokens")
            if not model_max:
                raise ValueError(
                    f"models.yaml entry for {self.model!r} is missing the "
                    f"required 'max_output_tokens' field. Every model must "
                    f"declare its maximum output token budget."
                )
            self.max_tokens = int(model_max)
        elif not self.max_tokens:
            raise ValueError(
                f"Model {self.model!r} is not registered in models.yaml and "
                f"no max_tokens is set. Add the model to models.yaml with a "
                f"'max_output_tokens' field."
            )
        # If model_id wasn't resolved, fall back to model (direct API id)
        if not self.model_id:
            self.model_id = self.model
        # Resolve API key from environment if not already set (needed on resume,
        # where provider is already populated so the block above is skipped)
        if not self.api_key:
            if resolved:
                self.api_key = os.environ.get(resolved["env_key"], "")
            if not self.api_key:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")


# Fields settable via config.yaml (workspace_dir, logs_dir, api_key excluded)
# max_tokens is intentionally excluded — it is derived from models.yaml only.
_YAML_CONFIG_FIELDS = frozenset(
    {
        "model",
        "verify_model",
        "max_iterations",
        "critic_every_n",
        "sympy_timeout_seconds",
        "max_tool_rounds",
        "progress_check_interval",
        "computation_token_alert",
        "tool_output_limit",
        "stall_threshold",
        "stall_recompute_limit",
        "min_er_for_completion",
        "api_retry_max",
        "api_retry_initial_delay",
        "api_retry_max_delay",
        "api_timeout",
        "budget_synthesis_margin",
        "orchestrator_comp_log_tail",
        "prior_failure_excerpt_chars",
        "max_open_rqs",
        "rq_evidence_cap",
        "max_refuted_retries",
        "auto_expire_iterations",
        "parse_retries",
        "max_tokens_retries",
        "pipeline_retry_max",
        "provider",
    }
)

# Fields persisted to config.json for resume (superset of _YAML_CONFIG_FIELDS)
_PERSIST_FIELDS = _YAML_CONFIG_FIELDS | {
    "model_id",
    "input_cost",
    "output_cost",
    "reasoning",
    "max_termination_retries",
}


def _resolve_model(model_key: str) -> dict | None:
    """Look up a model key in models.yaml, return {provider, model_id, env_key} or None."""
    path = Path(__file__).parent / "models.yaml"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            registry = yaml.safe_load(f)
        if not registry or not isinstance(registry, dict):
            return None
        entry = registry.get(model_key)
        if not entry or not isinstance(entry, dict):
            return None
        reasoning = {}
        for key in (
            "reasoning_budget",
            "reasoning_effort",
            "thinking_level",
            "thinking",
            "effort",
            "reasoning_format",
            "hf_provider",
            "timeout",
            "base_url",
            "tool_mode",
        ):
            if key in entry:
                reasoning[key] = entry[key]
        result = {
            "provider": entry["provider"],
            "model_id": entry.get("model_id", model_key),
            "env_key": entry.get("env_key", "ANTHROPIC_API_KEY"),
            "input_cost": float(entry.get("input_cost", 0)),
            "output_cost": float(entry.get("output_cost", 0)),
            "reasoning": reasoning,
        }
        if "max_output_tokens" in entry:
            result["max_output_tokens"] = int(entry["max_output_tokens"])
        return result
    except (OSError, yaml.YAMLError):
        return None


def load_config_yaml(path: Path) -> dict:
    """Load config from a YAML file, filtering to allowed fields."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or not isinstance(data, dict):
        return {}
    result = {}
    for key, value in data.items():
        if key in _YAML_CONFIG_FIELDS:
            result[key] = value
        else:
            warnings.warn(f"Unknown config key ignored: {key}")
    return result


def build_config(args: Namespace) -> Config:
    """Build Config with 3-tier precedence: CLI args > config.yaml > defaults."""
    kwargs: dict = {}

    # Layer 1: YAML config (if provided)
    if getattr(args, "config", None) is not None:
        kwargs.update(load_config_yaml(Path(args.config)))

    # Layer 2: CLI args override (only non-None values).
    # max_tokens is intentionally absent — derived from models.yaml.
    cli_fields = {
        "model",
        "max_iterations",
        "workspace_dir",
    }
    for field_name in cli_fields:
        cli_name = field_name.replace("-", "_")
        value = getattr(args, cli_name, None)
        if value is not None:
            kwargs[field_name] = value

    return Config(**kwargs)
