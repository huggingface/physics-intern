"""Tests for batch metadata merging in scripts.run_critpt_common."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_critpt_common import (  # noqa: E402
    RunResult,
    write_batch_metadata,
    write_initial_batch_metadata,
    load_resume_config,
    find_completed_submissions,
)


def _call(
    output_dir: Path, results: list[RunResult], *, start: datetime, end: datetime
) -> dict:
    write_batch_metadata(
        output_dir=output_dir,
        critpt_model="provider/model",
        all_results=results,
        generation_config={"model_key": "mk"},
        run_config={"timeout": 3600},
        start_time=start,
        end_time=end,
    )
    return json.loads((output_dir / "batch_metadata.json").read_text())


def _r(
    problem_n: int,
    *,
    success: bool,
    duration_s: float,
    stats: dict | None = None,
    error: str | None = None,
    workspace: Path | None = None,
) -> RunResult:
    return RunResult(
        problem_n=problem_n,
        problem_id=f"Challenge_{problem_n}_main",
        success=success,
        answer_code="code" if success else None,
        error=error,
        duration_s=duration_s,
        stats=stats,
        returncode=0 if success else 1,
        workspace_dir=workspace,
    )


T0 = datetime(2026, 4, 19, 10, 0, 0, tzinfo=timezone.utc)
T1 = T0 + timedelta(hours=1)
T2 = T1 + timedelta(minutes=30)
T3 = T2 + timedelta(minutes=10)


def test_no_prior_file_fresh_run(tmp_path):
    """With no prior metadata, output has no previous_attempts and cumulative==wall_clock."""
    r = _r(
        5,
        success=True,
        duration_s=200.0,
        stats={"cost_usd": 0.42, "input_tokens": 12000, "output_tokens": 3000},
    )
    data = _call(tmp_path, [r], start=T0, end=T0 + timedelta(seconds=210))

    assert len(data["problems"]) == 1
    p = data["problems"][0]
    assert p["problem_id"] == "Challenge_5_main"
    assert p["success"] is True
    assert "previous_attempts" not in p

    s = data["summary"]
    assert s["this_run_total"] == 1
    assert s["this_run_success"] == 1
    assert s["total_compute_s"] == 200.0
    assert s["wall_clock_s"] == 210.0
    assert s["cumulative_wall_clock_s"] == 210.0
    assert s["total_cost_usd"] == 0.42
    assert s["total_input_tokens"] == 12000


def test_overlap_moves_prior_into_previous_attempts(tmp_path):
    """Re-running a problem pushes the prior entry into previous_attempts."""
    prior = _r(
        5,
        success=False,
        duration_s=3600.0,
        error="timeout after 3600s",
        stats={"cost_usd": 1.10, "input_tokens": 40000, "output_tokens": 8000},
    )
    _call(tmp_path, [prior], start=T0, end=T1)

    current = _r(
        5,
        success=True,
        duration_s=200.0,
        stats={"cost_usd": 0.42, "input_tokens": 12000, "output_tokens": 3000},
    )
    data = _call(tmp_path, [current], start=T2, end=T2 + timedelta(seconds=210))

    assert len(data["problems"]) == 1
    p = data["problems"][0]
    assert p["success"] is True
    assert p["duration_s"] == 200.0
    attempts = p["previous_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["success"] is False
    assert attempts[0]["error"] == "timeout after 3600s"
    assert attempts[0]["duration_s"] == 3600.0
    assert attempts[0]["stats"]["input_tokens"] == 40000

    s = data["summary"]
    # Summary sums across current + every prior attempt for every problem.
    assert s["total_compute_s"] == pytest.approx(3800.0)
    assert s["total_cost_usd"] == pytest.approx(1.52, abs=1e-9)
    assert s["total_input_tokens"] == 52000
    assert s["total_output_tokens"] == 11000
    assert s["cumulative_wall_clock_s"] > s["wall_clock_s"]


def test_two_resume_cycles_flatten_previous_attempts(tmp_path):
    """Two resumes leave exactly two entries in previous_attempts (oldest first)."""
    attempt1 = _r(
        5,
        success=False,
        duration_s=3600.0,
        error="timeout 1",
        stats={"cost_usd": 1.0, "input_tokens": 30000, "output_tokens": 5000},
    )
    _call(tmp_path, [attempt1], start=T0, end=T1)

    attempt2 = _r(
        5,
        success=False,
        duration_s=3600.0,
        error="timeout 2",
        stats={"cost_usd": 1.2, "input_tokens": 35000, "output_tokens": 6000},
    )
    _call(tmp_path, [attempt2], start=T1, end=T2)

    attempt3 = _r(
        5,
        success=True,
        duration_s=200.0,
        stats={"cost_usd": 0.4, "input_tokens": 10000, "output_tokens": 2000},
    )
    data = _call(tmp_path, [attempt3], start=T2, end=T3)

    p = data["problems"][0]
    attempts = p["previous_attempts"]
    assert len(attempts) == 2
    assert attempts[0]["error"] == "timeout 1"
    assert attempts[1]["error"] == "timeout 2"

    s = data["summary"]
    assert s["total_compute_s"] == pytest.approx(7400.0)
    assert s["total_input_tokens"] == 75000
    assert s["total_output_tokens"] == 13000


def test_untouched_prior_problems_are_carried_through(tmp_path):
    """Prior problems absent from the current run survive verbatim."""
    r_a = _r(
        5,
        success=True,
        duration_s=100.0,
        stats={"cost_usd": 0.1, "input_tokens": 1000, "output_tokens": 500},
    )
    r_b = _r(
        7,
        success=False,
        duration_s=3600.0,
        error="timeout",
        stats={"cost_usd": 1.0, "input_tokens": 20000, "output_tokens": 4000},
    )
    _call(tmp_path, [r_a, r_b], start=T0, end=T1)

    # Resume only touches problem 7.
    r_b2 = _r(
        7,
        success=True,
        duration_s=300.0,
        stats={"cost_usd": 0.3, "input_tokens": 8000, "output_tokens": 1500},
    )
    data = _call(tmp_path, [r_b2], start=T1, end=T2)

    ids = [p["problem_id"] for p in data["problems"]]
    assert ids == ["Challenge_5_main", "Challenge_7_main"]  # sorted by problem_n

    p5 = next(p for p in data["problems"] if p["problem_id"] == "Challenge_5_main")
    # Untouched: carried through, no previous_attempts injected.
    assert p5["success"] is True
    assert p5.get("previous_attempts", []) == []

    p7 = next(p for p in data["problems"] if p["problem_id"] == "Challenge_7_main")
    assert p7["success"] is True
    assert len(p7["previous_attempts"]) == 1
    assert p7["previous_attempts"][0]["error"] == "timeout"

    s = data["summary"]
    # this_run only counts the current invocation (only problem 7 ran).
    assert s["this_run_total"] == 1
    # Cumulative sums over every attempt of every merged problem (5 + 7 current + 7 prior).
    assert s["total_compute_s"] == pytest.approx(100.0 + 300.0 + 3600.0)


def test_malformed_prior_json_is_treated_as_empty(tmp_path):
    """Corrupt prior metadata does not crash the writer; current run still writes cleanly."""
    (tmp_path / "batch_metadata.json").write_text("{not json")
    r = _r(
        5,
        success=True,
        duration_s=100.0,
        stats={"cost_usd": 0.1, "input_tokens": 1000, "output_tokens": 500},
    )
    data = _call(tmp_path, [r], start=T0, end=T1)
    assert len(data["problems"]) == 1
    assert "previous_attempts" not in data["problems"][0]


def test_summary_arithmetic_matches_sum_over_attempts(tmp_path):
    """Cumulative summary equals the sum over every attempt of every problem."""
    r_a1 = _r(
        5,
        success=False,
        duration_s=1000.0,
        error="x",
        stats={"cost_usd": 0.5, "input_tokens": 2000, "output_tokens": 500},
    )
    r_b1 = _r(
        7,
        success=True,
        duration_s=400.0,
        stats={"cost_usd": 0.3, "input_tokens": 1500, "output_tokens": 300},
    )
    _call(tmp_path, [r_a1, r_b1], start=T0, end=T1)

    r_a2 = _r(
        5,
        success=True,
        duration_s=200.0,
        stats={"cost_usd": 0.2, "input_tokens": 800, "output_tokens": 200},
    )
    data = _call(tmp_path, [r_a2], start=T1, end=T2)

    s = data["summary"]
    # Problem 5: 1000 + 200, Problem 7: 400 (carried through)
    assert s["total_compute_s"] == pytest.approx(1600.0)
    assert s["total_cost_usd"] == pytest.approx(1.0)
    assert s["total_input_tokens"] == 2000 + 800 + 1500
    assert s["total_output_tokens"] == 500 + 200 + 300


def _write_submission(output_dir: Path, n: int, *, with_code: bool = True) -> None:
    """Write a CritPt-style submission JSON for problem ``n``."""
    payload: dict = {"problem_id": f"Challenge_{n}_main"}
    if with_code:
        payload["generated_code"] = "x = 1"
    (output_dir / f"Challenge_{n}_main.json").write_text(json.dumps(payload))


def test_find_completed_submissions_returns_only_problems_with_code(tmp_path):
    """A submission counts as complete iff it has both ``problem_id`` and ``generated_code``."""
    _write_submission(tmp_path, 1)
    _write_submission(tmp_path, 3)
    _write_submission(tmp_path, 5, with_code=False)  # incomplete (no code)
    (tmp_path / "Challenge_7_main.json").write_text("{not json")  # corrupt
    (tmp_path / "logs").mkdir()  # ignored: not a submission
    (tmp_path / "batch_metadata.json").write_text("{}")  # ignored: wrong name

    assert find_completed_submissions(tmp_path) == {1, 3}


def test_find_completed_submissions_empty_dir_returns_empty(tmp_path):
    assert find_completed_submissions(tmp_path) == set()


def test_load_resume_config_missing_metadata_exits_cleanly(tmp_path):
    """The resume contract requires batch_metadata.json; missing → SystemExit, not crash."""
    with pytest.raises(SystemExit) as exc_info:
        load_resume_config(tmp_path)
    assert exc_info.value.code == 1


def test_initial_metadata_makes_killed_run_resumable(tmp_path):
    """Initial stub is a valid resume target and merges cleanly with later results."""
    write_initial_batch_metadata(
        output_dir=tmp_path,
        critpt_model="provider/model",
        generation_config={"model_key": "mk"},
        run_config={"timeout": 3600, "problems_subset": "1-3"},
        start_time=T0,
    )

    # Stub must be loadable by the resume path even with zero completed problems.
    gen_cfg, run_cfg = load_resume_config(tmp_path)
    assert gen_cfg["model_key"] == "mk"
    assert run_cfg["problems_subset"] == "1-3"

    stub = json.loads((tmp_path / "batch_metadata.json").read_text())
    assert stub["problems"] == []
    assert stub["summary"]["this_run_total"] == 0
    assert stub["summary"]["wall_clock_s"] == 0.0

    # End-of-run write merges results into the stub without losing configs.
    r = _r(
        5,
        success=True,
        duration_s=200.0,
        stats={"cost_usd": 0.4, "input_tokens": 10000, "output_tokens": 2000},
    )
    final = _call(tmp_path, [r], start=T1, end=T1 + timedelta(seconds=210))
    assert len(final["problems"]) == 1
    assert final["problems"][0]["problem_id"] == "Challenge_5_main"
    assert final["generation_config"]["model_key"] == "mk"


def _call_sibling(
    output_dir: Path,
    results: list[RunResult],
    *,
    start: datetime,
    end: datetime,
    include_sibling_history: bool = True,
) -> dict:
    """Variant of ``_call`` that opts into sibling-history folding."""
    write_batch_metadata(
        output_dir=output_dir,
        critpt_model="provider/model",
        all_results=results,
        generation_config={"model_key": "mk"},
        run_config={"timeout": 3600},
        start_time=start,
        end_time=end,
        include_sibling_history=include_sibling_history,
    )
    return json.loads((output_dir / "batch_metadata.json").read_text())


def test_sibling_history_folded_for_overlapping_problem(tmp_path):
    """Sibling's attempt on the same problem lands in previous_attempts."""
    sib = tmp_path / "20260418_100000"
    cur = tmp_path / "20260420_100000"
    sib.mkdir()
    cur.mkdir()

    _call(
        sib,
        [
            _r(
                5,
                success=False,
                duration_s=3600.0,
                error="timeout",
                stats={"cost_usd": 1.1, "input_tokens": 40000, "output_tokens": 8000},
            )
        ],
        start=T0,
        end=T1,
    )

    r = _r(
        5,
        success=True,
        duration_s=200.0,
        stats={"cost_usd": 0.42, "input_tokens": 12000, "output_tokens": 3000},
    )
    data = _call_sibling(cur, [r], start=T2, end=T2 + timedelta(seconds=210))

    assert len(data["problems"]) == 1
    p = data["problems"][0]
    attempts = p.get("previous_attempts", [])
    assert len(attempts) == 1
    assert attempts[0]["error"] == "timeout"
    assert attempts[0]["duration_s"] == 3600.0

    s = data["summary"]
    assert s["total_compute_s"] == pytest.approx(3800.0)
    assert s["total_cost_usd"] == pytest.approx(1.52, abs=1e-9)
    assert s["total_input_tokens"] == 52000


def test_multiple_siblings_ordered_by_timestamp(tmp_path):
    """Two siblings → previous_attempts ordered oldest-first."""
    sib_old = tmp_path / "20260418_100000"
    sib_new = tmp_path / "20260419_100000"
    cur = tmp_path / "20260420_100000"
    for d in (sib_old, sib_new, cur):
        d.mkdir()

    _call(
        sib_old,
        [
            _r(
                5,
                success=False,
                duration_s=1000.0,
                error="old",
                stats={"cost_usd": 0.5, "input_tokens": 1000, "output_tokens": 100},
            )
        ],
        start=T0,
        end=T0 + timedelta(minutes=10),
    )
    _call(
        sib_new,
        [
            _r(
                5,
                success=False,
                duration_s=2000.0,
                error="new",
                stats={"cost_usd": 0.6, "input_tokens": 2000, "output_tokens": 200},
            )
        ],
        start=T1,
        end=T1 + timedelta(minutes=20),
    )

    r = _r(
        5,
        success=True,
        duration_s=300.0,
        stats={"cost_usd": 0.1, "input_tokens": 500, "output_tokens": 50},
    )
    data = _call_sibling(cur, [r], start=T2, end=T2 + timedelta(seconds=310))

    attempts = data["problems"][0]["previous_attempts"]
    assert [a["error"] for a in attempts] == ["old", "new"]


def test_sibling_plus_same_dir_prior_no_duplication(tmp_path):
    """Same-dir prior sits AFTER sibling history, not duplicated."""
    sib = tmp_path / "20260418_100000"
    cur = tmp_path / "20260420_100000"
    sib.mkdir()
    cur.mkdir()

    _call(
        sib,
        [
            _r(
                5,
                success=False,
                duration_s=1000.0,
                error="sibling",
                stats={"cost_usd": 0.5, "input_tokens": 1000, "output_tokens": 100},
            )
        ],
        start=T0,
        end=T0 + timedelta(minutes=10),
    )

    # First run in cur, writes a prior entry in-dir (sibling history gets folded
    # already at this step, but there's no same-dir prior yet).
    _call_sibling(
        cur,
        [
            _r(
                5,
                success=False,
                duration_s=500.0,
                error="same-dir-prior",
                stats={"cost_usd": 0.2, "input_tokens": 500, "output_tokens": 50},
            )
        ],
        start=T1,
        end=T1 + timedelta(minutes=5),
    )

    # Second run in cur — now there's both a sibling AND a same-dir prior.
    r = _r(
        5,
        success=True,
        duration_s=300.0,
        stats={"cost_usd": 0.1, "input_tokens": 200, "output_tokens": 20},
    )
    data = _call_sibling(cur, [r], start=T2, end=T2 + timedelta(seconds=310))

    attempts = data["problems"][0]["previous_attempts"]
    # Expect: sibling (oldest), same-dir-prior (next). Exactly 2 entries — no duplication.
    assert len(attempts) == 2
    assert attempts[0]["error"] == "sibling"
    assert attempts[1]["error"] == "same-dir-prior"


def test_sibling_only_problem_not_carried_into_current_dir(tmp_path):
    """Option X: sibling-only problems never appear in current-dir metadata."""
    sib = tmp_path / "20260418_100000"
    cur = tmp_path / "20260420_100000"
    sib.mkdir()
    cur.mkdir()

    _call(
        sib,
        [
            _r(
                5,
                success=True,
                duration_s=100.0,
                stats={"cost_usd": 0.1, "input_tokens": 1000, "output_tokens": 100},
            ),
            _r(
                7,
                success=True,
                duration_s=200.0,
                stats={"cost_usd": 0.2, "input_tokens": 2000, "output_tokens": 200},
            ),
        ],
        start=T0,
        end=T1,
    )

    # Current run only touches problem 5.
    r = _r(
        5,
        success=True,
        duration_s=50.0,
        stats={"cost_usd": 0.05, "input_tokens": 500, "output_tokens": 50},
    )
    data = _call_sibling(cur, [r], start=T2, end=T2 + timedelta(seconds=60))

    ids = [p["problem_id"] for p in data["problems"]]
    assert ids == ["Challenge_5_main"], (
        "problem 7 from sibling must not be carried over"
    )
    assert data["num_submissions"] == 0  # no JSONs exist in cur yet


def test_malformed_sibling_metadata_skipped(tmp_path):
    """Corrupt sibling metadata is skipped; valid sibling still folds."""
    sib_bad = tmp_path / "20260417_100000"
    sib_good = tmp_path / "20260418_100000"
    cur = tmp_path / "20260420_100000"
    for d in (sib_bad, sib_good, cur):
        d.mkdir()
    (sib_bad / "batch_metadata.json").write_text("{not json")

    _call(
        sib_good,
        [
            _r(
                5,
                success=False,
                duration_s=1000.0,
                error="good-prior",
                stats={"cost_usd": 0.5, "input_tokens": 1000, "output_tokens": 100},
            )
        ],
        start=T0,
        end=T1,
    )

    r = _r(
        5,
        success=True,
        duration_s=200.0,
        stats={"cost_usd": 0.1, "input_tokens": 500, "output_tokens": 50},
    )
    data = _call_sibling(cur, [r], start=T2, end=T2 + timedelta(seconds=210))

    attempts = data["problems"][0]["previous_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["error"] == "good-prior"


def test_sibling_scan_leaves_cumulative_wall_clock_same_dir_only(tmp_path):
    """cumulative_wall_clock_s stays same-dir-only (no sibling pollution)."""
    sib = tmp_path / "20260418_100000"
    cur = tmp_path / "20260420_100000"
    sib.mkdir()
    cur.mkdir()

    _call(
        sib,
        [
            _r(
                5,
                success=False,
                duration_s=1000.0,
                error="x",
                stats={"cost_usd": 0.5, "input_tokens": 1000, "output_tokens": 100},
            )
        ],
        start=T0,
        end=T0 + timedelta(hours=2),
    )

    r = _r(
        5,
        success=True,
        duration_s=200.0,
        stats={"cost_usd": 0.1, "input_tokens": 500, "output_tokens": 50},
    )
    data = _call_sibling(cur, [r], start=T2, end=T2 + timedelta(seconds=210))

    s = data["summary"]
    # No same-dir prior and sibling wall clock is NOT folded in.
    assert s["wall_clock_s"] == pytest.approx(210.0)
    assert s["cumulative_wall_clock_s"] == s["wall_clock_s"]


def test_include_sibling_history_false_matches_today_behavior(tmp_path):
    """With the flag off, a sibling dir's presence does not change output."""
    sib = tmp_path / "20260418_100000"
    cur = tmp_path / "20260420_100000"
    sib.mkdir()
    cur.mkdir()

    _call(
        sib,
        [
            _r(
                5,
                success=False,
                duration_s=1000.0,
                error="sibling",
                stats={"cost_usd": 0.5, "input_tokens": 1000, "output_tokens": 100},
            )
        ],
        start=T0,
        end=T1,
    )

    r = _r(
        5,
        success=True,
        duration_s=200.0,
        stats={"cost_usd": 0.1, "input_tokens": 500, "output_tokens": 50},
    )
    # Default _call passes include_sibling_history unset → False.
    data = _call(cur, [r], start=T2, end=T2 + timedelta(seconds=210))

    p = data["problems"][0]
    assert "previous_attempts" not in p


def test_atomic_write_preserves_prior_on_replace_failure(tmp_path, monkeypatch):
    """If os.replace raises after the tmp file is written, the original survives."""
    r = _r(
        5,
        success=False,
        duration_s=100.0,
        error="x",
        stats={"cost_usd": 0.1, "input_tokens": 500, "output_tokens": 100},
    )
    _call(tmp_path, [r], start=T0, end=T1)
    original = (tmp_path / "batch_metadata.json").read_text()

    import run_critpt_common as rcc

    def boom(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(rcc.os, "replace", boom)

    r2 = _r(
        5,
        success=True,
        duration_s=200.0,
        stats={"cost_usd": 0.2, "input_tokens": 1000, "output_tokens": 300},
    )
    with pytest.raises(OSError):
        _call(tmp_path, [r2], start=T1, end=T2)

    assert (tmp_path / "batch_metadata.json").read_text() == original
