"""Tests for the independent verification / diagnosis script."""

from pathlib import Path


from open_dirac.verification import (
    WorkspaceContents,
    FormalEvalResult,
    load_workspace,
    load_reference_file,
    rerun_computations,
    run_formal_evaluation,
    write_formal_eval_report,
    load_or_run_formal_eval,
)
from open_dirac.verification.diagnosis import (
    DiagnosisEvent,
    DiagnosisResult,
    build_diagnosis_prompt,
    parse_diagnosis,
    write_diagnosis_report,
)
from open_dirac.verification.event_summary import (
    summarize_event_log as _summarize_event_log,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RESEARCH_STATE = """---
iteration: 10
established_results: 3
---

# Research State

## Working Hypotheses (WH) and Established Results (ER)

### ER-001: Schwarzschild metric
The line element is ds² = -(1-2M/r)dt² + (1-2M/r)⁻¹dr² + r²dΩ²

### ER-002: Surface gravity
κ = 1/(4M)

### ER-003: Hawking temperature
T_H = ℏκ/(2πk_B) = ℏ/(8πk_B M)
"""

CURRENT_TASK_TERMINATED = """---
task_type: terminate
reason: Research complete
iteration: 10
---

Research objectives achieved.
"""

CURRENT_TASK_NOT_TERMINATED = """---
task_type: research
iteration: 5
---

Continue investigating.
"""

CRITIQUE_LOG = """---
total_critiques: 1
---

# Active Critiques

## CRIT-001 [LOW] [UNRESOLVED]
- **Target:** ER-001
- **Filed:** iteration 4

### Phase 1: Reproduce
Standard Schwarzschild derivation.

### Phase 2: Objection
- **What is wrong:** Minor: signature convention not stated explicitly.

# Resolved Critiques
"""

EVIDENCE_LOG = """---
total_computations: 1
---

## Computation 1 — iteration 3
Verified ER-002 surface gravity numerically.
Verdict: VERIFIED
"""

WELL_FORMED_DIAGNOSIS_RESPONSE = """
Let me analyze this research run.

<diagnosis_summary>
The research run was largely successful, arriving at the correct Hawking temperature
through a standard derivation path. One notable error occurred in iteration 4 where
the researcher introduced a sign error in the surface gravity calculation, but this
was caught by the reviewer in iteration 6 and corrected by iteration 7.
</diagnosis_summary>

<chains>
EVENT-001 [CAUGHT] (iterations 4-7)
Agents: researcher, reviewer
Sign error in surface gravity κ — researcher wrote κ = -1/(4M) instead of κ = 1/(4M).
Reviewer flagged the sign issue in the REFUTED verdict.
Root cause: researcher applied Killing vector normalization with wrong sign convention.
Evidence: WH-002, CRIT-001, ER-002

EVENT-002 [PARTIAL] (iterations 8-10)
Agents: computer, orchestrator
Computation script for numerical verification timed out twice before succeeding on third attempt.
Root cause: initial script used symbolic integration instead of numerical evaluation.
Evidence: WH-003, ER-003
</chains>

<weakest_link>
The sign error in ER-002 was the closest call — if the reviewer had not caught it,
the final Hawking temperature would have been off by a sign.
</weakest_link>

<recommendations>
- Enforce sign convention checks in the surveyor's sanity checks section
- Add timeout handling for computation scripts to fail fast rather than stall
- Consider requiring numerical spot-checks before promoting WHs to ERs
</recommendations>
"""

PARTIAL_DIAGNOSIS_RESPONSE = """
The research had some issues.

<diagnosis_summary>
The system failed to arrive at the correct answer due to an uncaught error
in the surface gravity calculation.
</diagnosis_summary>
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_workspace(
    tmp_path,
    *,
    research_state=RESEARCH_STATE,
    current_task=CURRENT_TASK_TERMINATED,
    critique_log=CRITIQUE_LOG,
    evidence_log=EVIDENCE_LOG,
):
    """Create a mock workspace directory."""
    (tmp_path / "RESEARCH_STATE.md").write_text(research_state)
    (tmp_path / "CURRENT_TASK.md").write_text(current_task)
    (tmp_path / "CRITIQUE_LOG.md").write_text(critique_log)
    (tmp_path / "EVIDENCE_LOG.md").write_text(evidence_log)
    return str(tmp_path)


def test_load_workspace(tmp_path):
    ws_dir = _make_workspace(tmp_path)
    contents = load_workspace(ws_dir)

    assert contents.research_state == RESEARCH_STATE
    assert contents.critique_log == CRITIQUE_LOG
    assert contents.evidence_log == EVIDENCE_LOG
    assert contents.terminated_cleanly is True
    assert contents.frontmatter.get("task_type") == "terminate"


def test_load_workspace_not_terminated(tmp_path):
    ws_dir = _make_workspace(tmp_path, current_task=CURRENT_TASK_NOT_TERMINATED)
    contents = load_workspace(ws_dir)

    assert contents.terminated_cleanly is False


def test_load_workspace_missing_files(tmp_path):
    """Gracefully handle missing workspace files."""
    # Only create RESEARCH_STATE
    (tmp_path / "RESEARCH_STATE.md").write_text(RESEARCH_STATE)
    contents = load_workspace(str(tmp_path))

    assert contents.research_state == RESEARCH_STATE
    assert contents.critique_log == ""
    assert contents.evidence_log == ""
    assert contents.current_task == ""
    assert contents.terminated_cleanly is False


def test_load_workspace_with_scripts(tmp_path):
    ws_dir = _make_workspace(tmp_path)
    comp_dir = tmp_path / "computations"
    comp_dir.mkdir()
    (comp_dir / "check_001.py").write_text("print('ok')")
    (comp_dir / "check_002.py").write_text("print('ok')")

    contents = load_workspace(ws_dir)
    assert len(contents.computation_scripts) == 2


def test_load_workspace_loads_metrics(tmp_path):
    """METRICS.md is always loaded."""
    ws_dir = _make_workspace(tmp_path)
    (tmp_path / "METRICS.md").write_text("# Metrics\ntotal: 100")

    contents = load_workspace(ws_dir)
    assert "total: 100" in contents.metrics_md


def test_load_workspace_loads_event_log(tmp_path):
    """EVENT_LOG.jsonl is always loaded."""
    ws_dir = _make_workspace(tmp_path)
    event_data = '{"kind":"llm_call","agent":"orchestrator"}\n'
    (tmp_path / "EVENT_LOG.jsonl").write_text(event_data)

    contents = load_workspace(ws_dir)
    assert "llm_call" in contents.event_log


def test_load_workspace_no_metrics(tmp_path):
    """Missing METRICS.md leaves field empty."""
    ws_dir = _make_workspace(tmp_path)
    contents = load_workspace(ws_dir)
    assert contents.metrics_md == ""


def test_parse_diagnosis_well_formed():
    """All XML tags parsed correctly from a well-formed response."""
    formal_eval = FormalEvalResult(correct=True, method="simplify")
    result = parse_diagnosis(WELL_FORMED_DIAGNOSIS_RESPONSE, formal_eval=formal_eval)

    assert result.formal_outcome == "CORRECT"
    assert result.diagnosis_mode == "success_analysis"
    assert "largely successful" in result.summary.lower()
    assert len(result.events) == 2

    # Check first event
    ev0 = result.events[0]
    assert ev0.event_id == "EVENT-001"
    assert ev0.classification == "CAUGHT"
    assert ev0.chain_type == "correction_chain"
    assert "researcher" in ev0.agents_involved
    assert "reviewer" in ev0.agents_involved
    assert "sign" in ev0.description.lower()
    assert "WH-002" in ev0.evidence_ids

    # Check second event
    ev1 = result.events[1]
    assert ev1.event_id == "EVENT-002"
    assert ev1.classification == "PARTIAL"
    assert ev1.chain_type == "failure_chain"

    assert "sign error" in result.weakest_link.lower()
    assert len(result.recommendations) == 3
    assert len(result.parse_warnings) == 0


def test_parse_diagnosis_partial():
    """Missing tags should produce warnings, not crashes."""
    formal_eval = FormalEvalResult(correct=False, method="simplify")
    result = parse_diagnosis(PARTIAL_DIAGNOSIS_RESPONSE, formal_eval=formal_eval)

    assert result.formal_outcome == "INCORRECT"
    assert result.diagnosis_mode == "failure_analysis"
    assert "failed" in result.summary.lower()
    # Missing tags should generate warnings
    assert any("chains" in w.lower() for w in result.parse_warnings)
    assert any("weakest_link" in w.lower() for w in result.parse_warnings)
    assert any("recommendations" in w.lower() for w in result.parse_warnings)


def test_parse_diagnosis_empty():
    """Empty response should produce warnings."""
    result = parse_diagnosis("")

    assert result.formal_outcome == "SKIPPED"  # no formal_eval passed
    assert result.diagnosis_mode == "failure_analysis"
    assert len(result.parse_warnings) > 0


def test_build_diagnosis_prompt_includes_all_files(tmp_path):
    ws_dir = _make_workspace(tmp_path)
    contents = load_workspace(ws_dir)
    system, user_content = build_diagnosis_prompt(contents)

    assert "RESEARCH_STATE.md" in user_content
    assert "EVIDENCE_LOG.md" in user_content
    assert "CRITIQUE_LOG.md" in user_content
    assert "terminated cleanly" in user_content
    assert "ER-001" in user_content
    # System prompt should be the diagnosis prompt
    assert "diagnostic analyst" in system.lower()


def test_build_diagnosis_prompt_correct_framing(tmp_path):
    """CORRECT formal eval should frame as success analysis."""
    ws_dir = _make_workspace(tmp_path)
    contents = load_workspace(ws_dir)
    formal_eval = FormalEvalResult(correct=True, method="simplify")

    _, user_content = build_diagnosis_prompt(contents, formal_eval=formal_eval)

    assert "CORRECT" in user_content
    assert "correction chains" in user_content


def test_build_diagnosis_prompt_incorrect_framing(tmp_path):
    """INCORRECT formal eval should frame as failure analysis."""
    ws_dir = _make_workspace(tmp_path)
    contents = load_workspace(ws_dir)
    formal_eval = FormalEvalResult(correct=False, method="simplify")

    _, user_content = build_diagnosis_prompt(contents, formal_eval=formal_eval)

    assert "INCORRECT" in user_content
    assert "failure chains" in user_content


def test_build_diagnosis_prompt_skipped_framing(tmp_path):
    """Skipped formal eval should frame as no ground truth."""
    ws_dir = _make_workspace(tmp_path)
    contents = load_workspace(ws_dir)

    _, user_content = build_diagnosis_prompt(contents, formal_eval=None)

    assert "SKIPPED" in user_content
    assert "No ground truth" in user_content


def test_build_diagnosis_prompt_not_terminated(tmp_path):
    ws_dir = _make_workspace(tmp_path, current_task=CURRENT_TASK_NOT_TERMINATED)
    contents = load_workspace(ws_dir)
    _, user_content = build_diagnosis_prompt(contents)

    assert "did NOT terminate cleanly" in user_content


def test_build_diagnosis_prompt_with_known_answer(tmp_path):
    ws_dir = _make_workspace(tmp_path)
    contents = load_workspace(ws_dir)

    _, user_content = build_diagnosis_prompt(contents, known_answer="0.7687")
    assert "Known Answer" in user_content
    assert "0.7687" in user_content


def test_build_diagnosis_prompt_without_known_answer(tmp_path):
    ws_dir = _make_workspace(tmp_path)
    contents = load_workspace(ws_dir)

    _, user_content = build_diagnosis_prompt(contents, known_answer=None)
    assert "Known Answer" not in user_content


def test_build_diagnosis_prompt_with_rerun_results(tmp_path):
    from open_dirac.utils.sandbox import ExecutionResult
    from open_dirac.verification import RerunResult

    ws_dir = _make_workspace(tmp_path)
    contents = load_workspace(ws_dir)

    rerun = [
        RerunResult(
            script_path="/tmp/check_001.py",
            execution=ExecutionResult(
                stdout="All checks passed\n", stderr="", returncode=0, timed_out=False
            ),
        ),
        RerunResult(
            script_path="/tmp/check_002.py",
            execution=ExecutionResult(
                stdout="", stderr="Error!", returncode=1, timed_out=False
            ),
        ),
    ]

    _, user_content = build_diagnosis_prompt(contents, rerun_results=rerun)
    assert "Re-run Results" in user_content
    assert "check_001.py" in user_content
    assert "SUCCESS" in user_content
    assert "FAILED" in user_content


def test_build_diagnosis_prompt_includes_metrics(tmp_path):
    """METRICS.md content should appear in user content."""
    ws_dir = _make_workspace(tmp_path)
    (tmp_path / "METRICS.md").write_text("# Metrics\ntotal_iterations: 10")

    contents = load_workspace(ws_dir)
    _, user_content = build_diagnosis_prompt(contents)

    assert "METRICS.md" in user_content
    assert "total_iterations" in user_content


def test_build_diagnosis_prompt_includes_git_log(tmp_path):
    """Git log should appear in user content when available."""
    ws_dir = _make_workspace(tmp_path)
    contents = load_workspace(ws_dir)
    contents.git_log = (
        "abc1234 iteration 1: orchestrator\ndef5678 iteration 2: researcher"
    )
    _, user_content = build_diagnosis_prompt(contents)

    assert "Git Log" in user_content
    assert "abc1234" in user_content


def test_terminated_cleanly_detection(tmp_path):
    """Frontmatter parsing correctly detects termination."""
    # Terminated
    ws_dir = _make_workspace(tmp_path, current_task=CURRENT_TASK_TERMINATED)
    contents = load_workspace(ws_dir)
    assert contents.terminated_cleanly is True

    # Not terminated — use a fresh tmp dir
    tmp2 = tmp_path / "ws2"
    tmp2.mkdir()
    ws_dir2 = _make_workspace(tmp2, current_task=CURRENT_TASK_NOT_TERMINATED)
    contents2 = load_workspace(ws_dir2)
    assert contents2.terminated_cleanly is False

    # Missing CURRENT_TASK
    tmp3 = tmp_path / "ws3"
    tmp3.mkdir()
    (tmp3 / "RESEARCH_STATE.md").write_text("# state")
    contents3 = load_workspace(str(tmp3))
    assert contents3.terminated_cleanly is False


def test_terminated_cleanly_via_research_state_status(tmp_path):
    """Detect clean termination from RESEARCH_STATE status even when
    CURRENT_TASK has a non-terminate task type (engine exit path 2)."""
    completed_state = """---
iteration: 8
status: completed
established_results: 3
---

# Research State
"""
    partially_complete_state = """---
iteration: 8
status: partially_complete
established_results: 1
---

# Research State
"""
    # status: completed + non-terminate task → should be clean
    ws1 = tmp_path / "ws_completed"
    ws1.mkdir()
    ws_dir1 = _make_workspace(
        ws1, current_task=CURRENT_TASK_NOT_TERMINATED, research_state=completed_state
    )
    contents1 = load_workspace(ws_dir1)
    assert contents1.terminated_cleanly is True

    # status: partially_complete + non-terminate task → should be clean
    ws2 = tmp_path / "ws_partial"
    ws2.mkdir()
    ws_dir2 = _make_workspace(
        ws2,
        current_task=CURRENT_TASK_NOT_TERMINATED,
        research_state=partially_complete_state,
    )
    contents2 = load_workspace(ws_dir2)
    assert contents2.terminated_cleanly is True

    # No status in RESEARCH_STATE + non-terminate task → still not clean
    ws3 = tmp_path / "ws_no_status"
    ws3.mkdir()
    ws_dir3 = _make_workspace(ws3, current_task=CURRENT_TASK_NOT_TERMINATED)
    contents3 = load_workspace(ws_dir3)
    assert contents3.terminated_cleanly is False


def test_rerun_computations(tmp_path):
    """Re-run a trivial computation script via sandbox."""
    _make_workspace(tmp_path)
    comp_dir = tmp_path / "computations"
    comp_dir.mkdir()
    (comp_dir / "trivial.py").write_text("print(2 + 2)")

    results = rerun_computations(str(tmp_path), timeout=10)
    assert len(results) == 1
    assert results[0].execution.returncode == 0
    assert "4" in results[0].execution.stdout


# ---------------------------------------------------------------------------
# Diagnosis report writing tests
# ---------------------------------------------------------------------------


def test_write_diagnosis_report(tmp_path):
    """Diagnosis sections are appended to VERIFICATION.md."""
    ws_dir = _make_workspace(tmp_path)
    report_path = tmp_path / "VERIFICATION.md"
    report_path.write_text(
        "---\nformal_answer: correct\n---\n\n# Verification Report\n\n## Formal Answer Evaluation: CORRECT\n"
    )

    diag_result = DiagnosisResult(
        formal_outcome="CORRECT",
        diagnosis_mode="success_analysis",
        summary="Clean run with one corrected error.",
        events=[
            DiagnosisEvent(
                event_id="EVENT-001",
                chain_type="correction_chain",
                classification="CAUGHT",
                agents_involved=["researcher", "reviewer"],
                iterations="iterations 4-7",
                description="Sign error caught by reviewer.",
                root_cause="Wrong sign convention.",
                evidence_ids=["WH-002", "ER-002"],
            )
        ],
        weakest_link="Sign error in surface gravity.",
        recommendations=["Add sign checks", "Require numerical verification"],
    )

    write_diagnosis_report(diag_result, ws_dir)

    report = report_path.read_text()
    # Frontmatter should have diagnosis_mode
    assert "diagnosis_mode: success_analysis" in report
    # Original formal_answer should still be there
    assert "formal_answer: correct" in report
    # Diagnosis sections should be present
    assert "Diagnosis: Success Analysis" in report
    assert "Clean run" in report
    assert "EVENT-001" in report
    assert "CAUGHT" in report
    assert "Sign error" in report
    assert "Weakest Link" in report
    assert "Recommendations" in report
    assert "Add sign checks" in report


def test_write_diagnosis_report_creates_if_missing(tmp_path):
    """Should create VERIFICATION.md if it doesn't exist."""
    ws_dir = _make_workspace(tmp_path)
    report_path = tmp_path / "VERIFICATION.md"
    assert not report_path.exists()

    diag_result = DiagnosisResult(
        formal_outcome="INCORRECT",
        diagnosis_mode="failure_analysis",
        summary="Root cause: uncaught sign error.",
    )

    write_diagnosis_report(diag_result, ws_dir)

    report = report_path.read_text()
    assert "diagnosis_mode: failure_analysis" in report
    assert "Diagnosis: Failure Analysis" in report


# ---------------------------------------------------------------------------
# Formal eval report writing tests
# ---------------------------------------------------------------------------


def test_write_formal_eval_report_correct(tmp_path):
    """Correct formal eval writes proper VERIFICATION.md."""
    ws_dir = str(tmp_path)
    result = FormalEvalResult(correct=True, method="simplify", details="diff=0")
    write_formal_eval_report(result, ws_dir)

    report = (tmp_path / "VERIFICATION.md").read_text()
    assert "formal_answer: correct" in report
    assert "Formal Answer Evaluation: CORRECT" in report
    assert "simplify" in report


def test_write_formal_eval_report_incorrect(tmp_path):
    ws_dir = str(tmp_path)
    result = FormalEvalResult(correct=False, method="ratio_test")
    write_formal_eval_report(result, ws_dir)

    report = (tmp_path / "VERIFICATION.md").read_text()
    assert "formal_answer: incorrect" in report
    assert "INCORRECT" in report


def test_write_formal_eval_report_skipped(tmp_path):
    ws_dir = str(tmp_path)
    result = FormalEvalResult(skipped=True, skip_reason="No ANSWER.md")
    write_formal_eval_report(result, ws_dir)

    report = (tmp_path / "VERIFICATION.md").read_text()
    assert "formal_answer: skipped" in report
    assert "SKIPPED" in report
    assert "No ANSWER.md" in report


# ---------------------------------------------------------------------------
# Event log tests
# ---------------------------------------------------------------------------

EVENT_LOG_LINES = """\
{"kind":"scaffold","ts":"2026-03-13T14:00:00+00:00","iter":1,"category":"loop_control","event":"p1_budget_override","detail":"compute -> synthesize"}
{"kind":"llm_call","ts":"2026-03-13T14:00:01+00:00","agent":"orchestrator","iter":1,"model":"claude-sonnet-4-6","input_tokens":2000,"output_tokens":800,"stop_reason":"end_turn","duration_s":5.0,"system_prompt_chars":3000,"user_content_chars":1000,"response_chars":500,"reasoning_tokens":0,"answer_tokens":0,"round":0}
{"kind":"llm_call","ts":"2026-03-13T14:00:10+00:00","agent":"researcher","iter":2,"model":"claude-sonnet-4-6","input_tokens":5000,"output_tokens":2000,"stop_reason":"end_turn","duration_s":8.0,"system_prompt_chars":3000,"user_content_chars":4000,"response_chars":1500,"reasoning_tokens":0,"answer_tokens":0,"round":0}
{"kind":"scaffold","ts":"2026-03-13T14:00:20+00:00","iter":3,"category":"call_reliability","event":"forced_final_call","detail":"max_rounds"}
{"kind":"scaffold","ts":"2026-03-13T14:00:25+00:00","iter":3,"category":"call_reliability","event":"api_retry","detail":"attempt=1/3, TimeoutError"}
"""


def test_summarize_event_log_llm_table():
    """LLM call summary table is generated."""
    summary = _summarize_event_log(EVENT_LOG_LINES)
    assert "orchestrator" in summary
    assert "researcher" in summary
    assert "LLM Calls by Agent" in summary
    assert "2,000" in summary  # orchestrator input_tokens


def test_summarize_event_log_scaffold_categories():
    """Scaffold events are grouped by category."""
    summary = _summarize_event_log(EVENT_LOG_LINES)
    assert "call_reliability" in summary or "loop_control" in summary
    assert "p1_budget_override" in summary


def test_summarize_event_log_key_timeline():
    """Key events appear in timeline."""
    summary = _summarize_event_log(EVENT_LOG_LINES)
    assert "forced_final_call" in summary
    assert "api_retry" in summary


def test_summarize_event_log_empty():
    """Empty input returns empty string."""
    assert _summarize_event_log("") == ""
    assert _summarize_event_log("   \n  ") == ""


def test_summarize_event_log_truncation():
    """Summary is capped at max_chars."""
    # Generate many events
    lines = []
    for i in range(500):
        lines.append(
            f'{{"kind":"scaffold","ts":"T","iter":{i},"category":"call_reliability","event":"api_retry","detail":"attempt={i}"}}'
        )
    raw = "\n".join(lines)
    summary = _summarize_event_log(raw, max_chars=1000)
    assert len(summary) <= 1000


def test_build_diagnosis_prompt_includes_event_log(tmp_path):
    """Event log summary appears in the diagnosis prompt."""
    ws_dir = _make_workspace(tmp_path)
    (tmp_path / "EVENT_LOG.jsonl").write_text(EVENT_LOG_LINES)

    contents = load_workspace(ws_dir)
    _, user_content = build_diagnosis_prompt(contents)

    assert "Event Log Summary" in user_content
    assert "orchestrator" in user_content
    assert "LLM Calls by Agent" in user_content


# ---------------------------------------------------------------------------
# Formal answer evaluation tests
# ---------------------------------------------------------------------------

HAWKING_PROBLEM_DEF = {
    "answer": "T_H = hbar * c**3 / (8 * sp.pi * G * M * k_B)\n",
    "answer_template": (
        "import sympy as sp\n\n"
        "hbar, c, G, M, k_B = sp.symbols('hbar c G M k_B', positive=True)\n\n"
        "def answer(hbar, c, G, M, k_B):\n"
        "    T_H = ...\n"
        "    return T_H\n"
    ),
}

CORRECT_ANSWER_MD = (
    "import sympy as sp\n\n"
    "hbar, c, G, M, k_B = sp.symbols('hbar c G M k_B', positive=True)\n\n"
    "def answer(hbar, c, G, M, k_B):\n"
    "    T_H = hbar * c**3 / (8 * sp.pi * G * M * k_B)\n"
    "    return T_H\n"
)

INCORRECT_ANSWER_MD = (
    "import sympy as sp\n\n"
    "hbar, c, G, M, k_B = sp.symbols('hbar c G M k_B', positive=True)\n\n"
    "def answer(hbar, c, G, M, k_B):\n"
    "    T_H = hbar * c**3 / (4 * sp.pi * G * M * k_B)\n"
    "    return T_H\n"
)


def test_formal_eval_correct_answer(tmp_path):
    """Correct ANSWER.md → correct=True."""
    ws_dir = _make_workspace(tmp_path)
    (tmp_path / "ANSWER.md").write_text(CORRECT_ANSWER_MD)

    result = run_formal_evaluation(ws_dir, HAWKING_PROBLEM_DEF)

    assert not result.skipped
    assert result.correct is True
    assert result.method  # should report which method succeeded


def test_formal_eval_incorrect_answer(tmp_path):
    """Wrong coefficient → correct=False."""
    ws_dir = _make_workspace(tmp_path)
    (tmp_path / "ANSWER.md").write_text(INCORRECT_ANSWER_MD)

    result = run_formal_evaluation(ws_dir, HAWKING_PROBLEM_DEF)

    assert not result.skipped
    assert result.correct is False


def test_formal_eval_skip_no_problem(tmp_path):
    """No problem_def → skipped."""
    ws_dir = _make_workspace(tmp_path)
    (tmp_path / "ANSWER.md").write_text(CORRECT_ANSWER_MD)

    result = run_formal_evaluation(ws_dir, None)

    assert result.skipped
    assert "No problem" in result.skip_reason


def test_formal_eval_skip_no_answer(tmp_path):
    """Problem without answer field → skipped."""
    ws_dir = _make_workspace(tmp_path)
    (tmp_path / "ANSWER.md").write_text(CORRECT_ANSWER_MD)

    result = run_formal_evaluation(ws_dir, {"problem": "something"})

    assert result.skipped
    assert "No problem" in result.skip_reason


def test_formal_eval_skip_no_template(tmp_path):
    """Problem with answer but no answer_template → skipped."""
    ws_dir = _make_workspace(tmp_path)
    (tmp_path / "ANSWER.md").write_text(CORRECT_ANSWER_MD)

    result = run_formal_evaluation(ws_dir, {"answer": "42"})

    assert result.skipped
    assert "answer_template" in result.skip_reason


def test_formal_eval_skip_no_answer_md(tmp_path):
    """ANSWER.md missing → skipped."""
    ws_dir = _make_workspace(tmp_path)
    # No ANSWER.md written

    result = run_formal_evaluation(ws_dir, HAWKING_PROBLEM_DEF)

    assert result.skipped
    assert "ANSWER.md" in result.skip_reason


def test_formal_eval_already_fenced(tmp_path):
    """ANSWER.md already has fences → should not double-fence."""
    ws_dir = _make_workspace(tmp_path)
    fenced = f"```python\n{CORRECT_ANSWER_MD}\n```"
    (tmp_path / "ANSWER.md").write_text(fenced)

    result = run_formal_evaluation(ws_dir, HAWKING_PROBLEM_DEF)

    assert not result.skipped
    assert result.correct is True


def test_formal_eval_prompt_inclusion(tmp_path):
    """Formal eval result is included in the diagnosis prompt."""
    ws_dir = _make_workspace(tmp_path)
    contents = load_workspace(ws_dir)

    correct_eval = FormalEvalResult(correct=True, method="simplify", details="diff=0")
    _, user_content = build_diagnosis_prompt(contents, formal_eval=correct_eval)

    assert "Formal Answer Evaluation" in user_content
    assert "CORRECT" in user_content
    assert "simplify" in user_content


def test_formal_eval_prompt_skipped_not_included(tmp_path):
    """Skipped formal eval should show SKIPPED framing."""
    ws_dir = _make_workspace(tmp_path)
    contents = load_workspace(ws_dir)

    skipped_eval = FormalEvalResult(skipped=True, skip_reason="No ANSWER.md")
    _, user_content = build_diagnosis_prompt(contents, formal_eval=skipped_eval)

    assert "SKIPPED" in user_content
    assert "No ground truth" in user_content


# ---------------------------------------------------------------------------
# Reference file loading
# ---------------------------------------------------------------------------


def test_load_reference_file_with_python_tag(tmp_path, monkeypatch):
    """Reference file with ```python tag → extracts answer expression."""
    monkeypatch.setattr("open_dirac.verification.workspace.REFERENCES_DIR", tmp_path)
    ref = tmp_path / "my_problem.md"
    ref.write_text("```python\ndelta = 3 * x + y\n```\n\n# Typical Good Run\n...")

    answer, content = load_reference_file(Path("problems/my_problem.yaml"))

    assert answer == "delta = 3 * x + y"
    assert "Typical Good Run" in content


def test_load_reference_file_without_tag(tmp_path, monkeypatch):
    """Reference file with bare ``` block → still extracts answer."""
    monkeypatch.setattr("open_dirac.verification.workspace.REFERENCES_DIR", tmp_path)
    ref = tmp_path / "my_problem.md"
    ref.write_text("```\nF = 1 - p**2\n```\n\n# Run description")

    answer, content = load_reference_file(Path("problems/my_problem.yaml"))

    assert answer == "F = 1 - p**2"


def test_load_reference_file_not_found(tmp_path, monkeypatch):
    """No matching reference file → (None, None)."""
    monkeypatch.setattr("open_dirac.verification.workspace.REFERENCES_DIR", tmp_path)

    answer, content = load_reference_file(Path("problems/nonexistent.yaml"))

    assert answer is None
    assert content is None


def test_load_reference_file_none_path():
    """None problem path → (None, None)."""
    answer, content = load_reference_file(None)

    assert answer is None
    assert content is None


def test_load_reference_file_no_code_block(tmp_path, monkeypatch):
    """Reference file without code block → answer is None, content is returned."""
    monkeypatch.setattr("open_dirac.verification.workspace.REFERENCES_DIR", tmp_path)
    ref = tmp_path / "my_problem.md"
    ref.write_text("# Just a description\nNo code block here.")

    answer, content = load_reference_file(Path("problems/my_problem.yaml"))

    assert answer is None
    assert "Just a description" in content


# ---------------------------------------------------------------------------
# Formal eval fallback to reference file
# ---------------------------------------------------------------------------


def test_formal_eval_fallback_to_reference(tmp_path, monkeypatch):
    """Empty answer in YAML + reference file with answer → formal eval proceeds."""
    ws_dir = _make_workspace(tmp_path)
    (tmp_path / "ANSWER.md").write_text(CORRECT_ANSWER_MD)

    # Problem def with empty answer but valid template
    problem_def = dict(HAWKING_PROBLEM_DEF)
    problem_def["answer"] = ""

    # Mock reference file to return the correct answer
    ref_answer = HAWKING_PROBLEM_DEF["answer"]
    monkeypatch.setattr(
        "open_dirac.verification.formal_eval.load_reference_file",
        lambda path: (ref_answer, "# reference content"),
    )

    result = run_formal_evaluation(ws_dir, problem_def, problem_path=Path("test.yaml"))

    assert not result.skipped
    assert result.correct is True


def test_formal_eval_no_fallback_when_answer_present(tmp_path, monkeypatch):
    """YAML has answer → reference file is not consulted."""
    ws_dir = _make_workspace(tmp_path)
    (tmp_path / "ANSWER.md").write_text(CORRECT_ANSWER_MD)

    # Track whether load_reference_file was called
    called = []
    monkeypatch.setattr(
        "open_dirac.verification.formal_eval.load_reference_file",
        lambda path: (called.append(1), None) or (None, None),
    )

    result = run_formal_evaluation(
        ws_dir, HAWKING_PROBLEM_DEF, problem_path=Path("test.yaml")
    )

    assert not result.skipped
    assert result.correct is True
    assert len(called) == 0  # reference file not consulted


# ---------------------------------------------------------------------------
# build_diagnosis_prompt with reference content
# ---------------------------------------------------------------------------


def test_build_diagnosis_prompt_with_reference_content():
    """Reference content is included in diagnosis prompt."""
    contents = WorkspaceContents(
        workspace_dir="/tmp/test",
        research_state="# State",
        evidence_log="# Evidence",
        critique_log="# Critiques",
    )
    ref_content = "# Typical Good Run\nExpected: 5 iterations, VALID verdict."

    _, user_content = build_diagnosis_prompt(contents, reference_content=ref_content)

    assert "## Reference Document" in user_content
    assert "Typical Good Run" in user_content
    assert "Expected: 5 iterations" in user_content


def test_build_diagnosis_prompt_without_reference_content():
    """No reference content → no Reference Document section."""
    contents = WorkspaceContents(
        workspace_dir="/tmp/test",
        research_state="# State",
        evidence_log="# Evidence",
        critique_log="# Critiques",
    )

    _, user_content = build_diagnosis_prompt(contents, reference_content=None)

    assert "Reference Document" not in user_content


# ---------------------------------------------------------------------------
# load_or_run_formal_eval
# ---------------------------------------------------------------------------


def test_load_or_run_reads_existing_report(tmp_path):
    """If VERIFICATION.md has formal_answer, read it instead of re-running."""
    _make_workspace(tmp_path)
    (tmp_path / "VERIFICATION.md").write_text(
        "---\nformal_answer: correct\n---\n\n# Report\n"
    )

    result = load_or_run_formal_eval(str(tmp_path), None, None)

    assert result.correct is True
    assert result.method == "from_report"


def test_load_or_run_falls_back_to_fresh(tmp_path):
    """If no VERIFICATION.md, run formal eval fresh."""
    _make_workspace(tmp_path)
    (tmp_path / "ANSWER.md").write_text(CORRECT_ANSWER_MD)

    result = load_or_run_formal_eval(str(tmp_path), HAWKING_PROBLEM_DEF, None)

    assert result.correct is True
    assert result.method != "from_report"
