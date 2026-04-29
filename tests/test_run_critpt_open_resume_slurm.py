"""Integration checks for ``serve/run_critpt_open_resume.slurm`` (no Slurm).

The Slurm script waits for ``endpoint.env``, polls ``/health``, then launches
CritPt. We validate the bash flow with a local HTTP server and
``CRITPT_RESUME_DRY_RUN=1`` so no GPU or batch run is required.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "serve" / "run_critpt_open_resume.slurm"


class _HealthHandler(BaseHTTPRequestHandler):
    """Reply 200 for ``GET /health`` (vLLM-style liveness)."""

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        p = self.path.split("?", 1)[0].rstrip("/")
        if p == "" or p == "/health":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def _health_server(port: int) -> Generator[None, None, None]:
    server = ThreadingHTTPServer(("127.0.0.1", port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def _write_endpoint_env(job_id: str, port: int) -> Path:
    log_dir = _REPO_ROOT / "serve" / "logs" / "vllm" / job_id
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "endpoint.env"
    text = (
        f"MODEL=test-model\n"
        f"SERVED_MODEL_NAME=test-model\n"
        f"HEAD_NODE=localhost\n"
        f"HEAD_IP=127.0.0.1\n"
        f"PORT={port}\n"
        f"BASE_URL=http://127.0.0.1:{port}/v1\n"
    )
    path.write_text(text, encoding="utf-8")
    return log_dir


def _run_resume_script(*, tmp_home: Path, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(tmp_home)
    env["CRITPT_SKIP_BASHRC"] = "1"
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(_SCRIPT)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.fixture
def isolated_home(tmp_path: Path) -> Path:
    return tmp_path / "home"


def test_resume_slurm_dry_run_after_health_ok(isolated_home: Path) -> None:
    """endpoint.env present, /health responds → dry run exits 0 without CritPt."""
    port = _free_port()
    job_id = f"pytest_resume_{uuid.uuid4().hex[:12]}"
    log_dir = _write_endpoint_env(job_id, port)
    try:
        with _health_server(port):
            proc = _run_resume_script(
                tmp_home=isolated_home,
                extra_env={
                    "SERVE_JOB": job_id,
                    "RESUME_DIR": str(isolated_home / "resume"),
                    "CRITPT_RESUME_DRY_RUN": "1",
                    "CRITPT_HEALTH_POLL_SLEEP": "1",
                },
            )
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "vLLM ready" in proc.stdout
    assert "CRITPT_RESUME_DRY_RUN=1" in proc.stdout


def test_resume_slurm_health_timeout(isolated_home: Path) -> None:
    """endpoint.env present but nothing serves /health → script exits 1."""
    port = _free_port()
    job_id = f"pytest_resume_ht_{uuid.uuid4().hex[:12]}"
    log_dir = _write_endpoint_env(job_id, port)
    try:
        proc = _run_resume_script(
            tmp_home=isolated_home,
            extra_env={
                "SERVE_JOB": job_id,
                "RESUME_DIR": str(isolated_home / "resume"),
                "CRITPT_HEALTH_WAIT_SEC": "2",
                "CRITPT_HEALTH_POLL_SLEEP": "1",
            },
        )
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)

    assert proc.returncode == 1
    assert "health timeout" in proc.stdout + proc.stderr
