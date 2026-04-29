"""Regressions for serve/serve.slurm idle-watchdog Prometheus scraping logic.

The watchdog sums successful-request counter lines from GET /metrics. The awk
program must stay in sync with start_idle_watchdog() in serve/serve.slurm.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Mirrors serve/serve.slurm start_idle_watchdog curl|awk (keep in sync).
_WATCHDOG_SUM_AWK = """
/^vllm:request_success_total[ {]/ { sum += $NF; ok = 1 }
/^vllm_request_success_total[ {]/ { sum += $NF; ok = 1 }
END { if (ok) print sum }
"""


def _awk_sum(metrics_file: Path) -> str:
    proc = subprocess.run(
        ["awk", _WATCHDOG_SUM_AWK, str(metrics_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def test_watchdog_sums_colon_metric_lines() -> None:
    got = _awk_sum(_FIXTURES / "vllm_metrics_request_success.txt")
    assert got == "5"


def test_watchdog_sums_underscore_metric_lines() -> None:
    got = _awk_sum(_FIXTURES / "vllm_metrics_request_success_underscore.txt")
    assert got == "10"


def test_watchdog_mixed_naming_styles_sum(tmp_path: Path) -> None:
    """When both naming styles appear, awk sums all matching lines (defensive)."""
    combined = _FIXTURES / "vllm_metrics_request_success.txt"
    text = combined.read_text()
    text += 'vllm_request_success_total{finished_reason="abort",model_name="x"} 1\n'
    path = tmp_path / "m.txt"
    path.write_text(text)
    assert _awk_sum(path) == "6"


def test_watchdog_empty_metrics_no_print() -> None:
    proc = subprocess.run(
        ["awk", _WATCHDOG_SUM_AWK],
        check=True,
        input="# no counters\n",
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == ""
