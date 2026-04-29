"""Tests for config loading, YAML merge, and CLI parsers."""

import shlex
import subprocess
import sys
import warnings
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from open_dirac.config import (
    Config,
    DEFAULTS,
    _YAML_CONFIG_FIELDS,
    load_config_yaml,
    build_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = PROJECT_ROOT / "src" / "open_dirac" / "models.yaml"
RESOLVE_SERVE_CONFIG = PROJECT_ROOT / "serve" / "resolve_serve_config.py"


# ---------------------------------------------------------------------------
# DEFAULTS loaded from config.default.yaml
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_defaults_has_required_keys(self):
        # max_tokens is intentionally absent — it's sourced from models.yaml
        required = {
            "model",
            "verify_model",
            "max_iterations",
            "critic_every_n",
            "sympy_timeout_seconds",
            "max_tool_rounds",
            "tool_output_limit",
            "min_er_for_completion",
        }
        assert required.issubset(DEFAULTS.keys())

    def test_max_tokens_not_in_defaults(self):
        # max_tokens is derived from models.yaml, not config.default.yaml
        assert "max_tokens" not in DEFAULTS

    def test_defaults_model_is_string(self):
        assert isinstance(DEFAULTS["model"], str) and DEFAULTS["model"]

    def test_defaults_verify_model_is_string(self):
        assert isinstance(DEFAULTS["verify_model"], str) and DEFAULTS["verify_model"]

    def test_stall_recompute_limit_default(self):
        assert Config().stall_recompute_limit == 2

    def test_progress_check_interval_default(self):
        assert Config().progress_check_interval == 4
        assert DEFAULTS["progress_check_interval"] == 4

    def test_progress_check_interval_in_yaml_fields(self):
        assert "progress_check_interval" in _YAML_CONFIG_FIELDS


# ---------------------------------------------------------------------------
# load_config_yaml
# ---------------------------------------------------------------------------


class TestLoadConfigYaml:
    def test_valid_file(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({"model": "custom-model", "max_iterations": 10}))
        result = load_config_yaml(cfg)
        assert result == {"model": "custom-model", "max_iterations": 10}

    def test_unknown_keys_warn(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({"model": "x", "bogus_key": 42}))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_config_yaml(cfg)
            assert len(w) == 1
            assert "bogus_key" in str(w[0].message)
        assert "bogus_key" not in result
        assert result["model"] == "x"

    def test_empty_file(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        assert load_config_yaml(cfg) == {}

    def test_excluded_fields_rejected(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({"workspace_dir": "/tmp/ws", "api_key": "secret"}))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_config_yaml(cfg)
            assert len(w) == 2
        assert "workspace_dir" not in result
        assert "api_key" not in result

    def test_verify_model_accepted(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({"verify_model": "some-model"}))
        result = load_config_yaml(cfg)
        assert result["verify_model"] == "some-model"


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------


class TestBuildConfig:
    def test_defaults_only(self):
        args = Namespace(
            config=None,
            model=None,
            max_iterations=None,
            workspace_dir=None,
        )
        cfg = build_config(args)
        assert cfg.model == DEFAULTS["model"]
        assert cfg.verify_model == DEFAULTS["verify_model"]
        assert cfg.max_iterations == DEFAULTS["max_iterations"]
        # max_tokens resolved from models.yaml for the default model
        assert cfg.max_tokens > 0

    def test_cli_overrides_defaults(self):
        args = Namespace(
            config=None,
            model="claude-4.6-opus",
            max_iterations=5,
            workspace_dir=None,
            critic_every_n=None,
            sympy_timeout_seconds=None,
        )
        cfg = build_config(args)
        assert cfg.model == "claude-4.6-opus"
        assert cfg.max_iterations == 5
        # max_tokens tracks the model's declared max_output_tokens
        assert cfg.max_tokens == 128000

    def test_yaml_overrides_defaults(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.dump({"model": "claude-4.6-sonnet", "max_iterations": 50})
        )
        args = Namespace(
            config=str(cfg_file),
            model=None,
            max_iterations=None,
            workspace_dir=None,
        )
        cfg = build_config(args)
        assert cfg.model == "claude-4.6-sonnet"
        assert cfg.max_iterations == 50
        assert cfg.max_tokens == 65536  # from models.yaml

    def test_cli_overrides_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.dump({"model": "claude-4.6-sonnet", "max_iterations": 50})
        )
        args = Namespace(
            config=str(cfg_file),
            model="claude-4.6-opus",
            max_iterations=None,
            workspace_dir=None,
        )
        cfg = build_config(args)
        assert cfg.model == "claude-4.6-opus"
        assert cfg.max_iterations == 50  # from YAML
        assert cfg.max_tokens == 128000  # tracks CLI-selected model

    def test_max_tokens_not_settable_via_yaml(self, tmp_path):
        """max_tokens is ignored in config YAML (models.yaml is the source)."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.dump({"model": "claude-4.6-sonnet", "max_tokens": 8192})
        )
        args = Namespace(
            config=str(cfg_file),
            model=None,
            max_iterations=10,
            workspace_dir=None,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cfg = build_config(args)
        # Attempted override is ignored; value comes from models.yaml
        assert cfg.max_tokens == 65536
        assert cfg.max_iterations == 10


# ---------------------------------------------------------------------------
# Model registry resolution
# ---------------------------------------------------------------------------


class TestModelRegistryResolution:
    def test_glm_5_1_local_vllm_key_resolves(self):
        cfg = Config(model="zai-org/GLM-5.1")
        assert cfg.provider == "vllm"
        assert cfg.model_id == "zai-org/GLM-5.1"
        assert cfg.max_tokens == 131072
        assert cfg.reasoning["reasoning_format"] == "separate_field"
        assert cfg.reasoning["tool_mode"] == "api"

    def test_kimi_k2_6_local_vllm_key_resolves(self):
        cfg = Config(model="moonshotai/Kimi-K2.6")
        assert cfg.provider == "vllm"
        assert cfg.model_id == "moonshotai/Kimi-K2.6"
        assert cfg.max_tokens == 200000
        assert cfg.reasoning["reasoning_format"] == "separate_field"
        assert cfg.reasoning["tool_mode"] == "api"


# ---------------------------------------------------------------------------
# models.yaml `serve` block — consumed by serve/serve.slurm
# ---------------------------------------------------------------------------


class TestServeConfig:
    """The serve.slurm script reads `serve.{nodes,gpus_per_node,vllm_args}` for
    each model. These tests pin the contract for the huge local models so a
    drive-by yaml edit cannot silently break the launcher."""

    @pytest.fixture(scope="class")
    def registry(self) -> dict:
        return yaml.safe_load(MODELS_YAML.read_text())

    def test_glm_5_1_serve_block(self, registry):
        serve = registry["zai-org/GLM-5.1"]["serve"]
        assert serve["nodes"] == 3
        assert serve["gpus_per_node"] == 8
        args = " ".join(serve["vllm_args"])
        # DeepGEMM JIT cache/toolkit setup in serve.slurm makes CUDA graphs usable.
        assert "--enforce-eager" not in args
        assert "--safetensors-load-strategy prefetch" in args  # 17x load speedup
        assert "--trust-remote-code" in args

    def test_kimi_k2_6_serve_block(self, registry):
        serve = registry["moonshotai/Kimi-K2.6"]["serve"]
        assert serve["nodes"] == 4
        assert serve["gpus_per_node"] == 8
        assert serve["reasoning_parser"] == "kimi_k2"
        args = " ".join(serve["vllm_args"])
        assert "--enable-expert-parallel" in args
        assert "--enable-auto-tool-choice" in args
        assert "--tool-call-parser kimi_k2" in args
        assert "--safetensors-load-strategy prefetch" in args
        # Champion config explicitly does NOT use --enforce-eager (CUDA graphs
        # give the dominant 4x throughput win on Kimi).
        assert "--enforce-eager" not in args


# ---------------------------------------------------------------------------
# serve/resolve_serve_config.py — bridges models.yaml → serve.slurm shell vars
# ---------------------------------------------------------------------------


def _run_resolver(model: str) -> dict[str, str]:
    """Invoke resolve_serve_config.py and parse its KEY=value output."""
    result = subprocess.run(
        [sys.executable, str(RESOLVE_SERVE_CONFIG), model],
        capture_output=True,
        text=True,
        check=True,
    )
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        # Strip the array-form wrapper for DEFAULT_VLLM_ARGS, leave others quoted.
        if v.startswith("( ") and v.endswith(" )"):
            out[k] = v[2:-2]
        else:
            out[k] = shlex.split(v)[0] if v else ""
    return out


class TestResolveServeConfig:
    """resolve_serve_config.py emits the shell vars serve.slurm relies on. The
    `DEFAULT_MODEL_ID` line is new in this PR (used to support alias keys)."""

    def test_emits_all_required_shell_vars_for_kimi(self):
        out = _run_resolver("moonshotai/Kimi-K2.6")
        assert out["DEFAULT_MODEL_ID"] == "moonshotai/Kimi-K2.6"
        assert out["DEFAULT_NODES"] == "4"
        assert out["DEFAULT_GPUS_PER_NODE"] == "8"
        assert out["DEFAULT_REASONING_PARSER"] == "kimi_k2"
        assert "--enable-expert-parallel" in out["DEFAULT_VLLM_ARGS"]
        assert "--enable-auto-tool-choice" in out["DEFAULT_VLLM_ARGS"]
        assert "--tool-call-parser" in out["DEFAULT_VLLM_ARGS"]

    def test_emits_all_required_shell_vars_for_glm(self):
        out = _run_resolver("zai-org/GLM-5.1")
        assert out["DEFAULT_MODEL_ID"] == "zai-org/GLM-5.1"
        assert out["DEFAULT_NODES"] == "3"
        assert out["DEFAULT_GPUS_PER_NODE"] == "8"
        assert "--enforce-eager" not in out["DEFAULT_VLLM_ARGS"]
        assert "--safetensors-load-strategy" in out["DEFAULT_VLLM_ARGS"]

    def test_unknown_model_falls_back_to_input_as_model_id(self):
        """No registry entry → DEFAULT_MODEL_ID == input model, no nodes/args."""
        out = _run_resolver("not/a-real-model-key")
        assert out["DEFAULT_MODEL_ID"] == "not/a-real-model-key"
        assert out["DEFAULT_NODES"] == ""
        assert out["DEFAULT_VLLM_ARGS"] == ""


# ---------------------------------------------------------------------------
# main.py parser
# ---------------------------------------------------------------------------


class TestMainParser:
    def test_basic_args(self):
        from open_dirac.main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["problems/test.yaml", "--max-iterations", "5", "--model", "my-model"]
        )
        assert args.problem == Path("problems/test.yaml")
        assert args.max_iterations == 5
        assert args.model == "my-model"
        assert args.config is None

    def test_all_args(self):
        from open_dirac.main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "p.yaml",
                "--config",
                "c.yaml",
                "--model",
                "m",
                "--max-iterations",
                "3",
                "--workspace-dir",
                "/tmp/ws",
            ]
        )
        assert args.config == Path("c.yaml")
        assert args.workspace_dir == Path("/tmp/ws")

    def test_max_tokens_flag_removed(self):
        """--max-tokens is no longer a valid CLI flag."""
        from open_dirac.main import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["p.yaml", "--max-tokens", "1024"])

    def test_bad_int_exits(self):
        from open_dirac.main import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["p.yaml", "--max-iterations", "abc"])

    def test_defaults_are_none(self):
        from open_dirac.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["p.yaml"])
        assert args.model is None
        assert args.max_iterations is None
        assert args.workspace_dir is None


# ---------------------------------------------------------------------------
# verify.py parser
# ---------------------------------------------------------------------------


class TestVerifyParser:
    def test_basic_args(self):
        from open_dirac.verification.cli import build_verify_parser

        parser = build_verify_parser()
        args = parser.parse_args(["workspaces/run1"])
        assert args.workspace_dir == Path("workspaces/run1")
        assert args.model == DEFAULTS["verify_model"]

    def test_custom_model(self):
        from open_dirac.verification.cli import build_verify_parser

        parser = build_verify_parser()
        args = parser.parse_args(["ws", "--model", "opus"])
        assert args.model == "opus"

    def test_max_tokens_flag_removed(self):
        from open_dirac.verification.cli import build_verify_parser

        parser = build_verify_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ws", "--max-tokens", "8192"])
