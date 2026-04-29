"""Tests for per-call conversation logging in llm.py."""

import open_dirac.llm as llm_module
from open_dirac.config import Config
from open_dirac.llm import (
    AgentResult,
    LLMResponse,
    _render_agent_conversation_log,
    _write_agent_conversation_log,
    _write_conversation_log,
)
from open_dirac.workspace import WorkspaceManager


def _make_response(**overrides):
    defaults = dict(
        text="response text",
        input_tokens=100,
        output_tokens=50,
        stop_reason="end_turn",
        duration=1.23,
    )
    defaults.update(overrides)
    return LLMResponse(**defaults)


def test_creates_correctly_named_file(tmp_path):
    config = Config(logs_dir=str(tmp_path))
    # Reset seq counter for clean test
    llm_module._call_seq.clear()

    _write_conversation_log(
        config, _make_response(), "system prompt", "user msg", "researcher", 3
    )

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "iter003_01_researcher.md"


def test_file_contains_expected_sections(tmp_path):
    config = Config(logs_dir=str(tmp_path))
    llm_module._call_seq.clear()

    _write_conversation_log(
        config,
        _make_response(text="the answer"),
        "sys prompt here",
        "user content here",
        "orchestrator",
        5,
    )

    content = (tmp_path / "iter005_01_orchestrator.md").read_text()
    assert '<SYSTEM_PROMPT chars="15">' in content
    assert "sys prompt here" in content
    assert '<USER_MESSAGE chars="17">' in content
    assert "user content here" in content
    assert '<LLM_RESPONSE chars="10"' in content
    assert 'tokens_in="100"' in content
    assert 'tokens_out="50"' in content
    assert 'duration="1.2s"' in content
    assert 'stop="end_turn"' in content
    assert "the answer" in content
    # Header table was removed; log starts directly with system prompt
    assert content.startswith("<SYSTEM_PROMPT chars=")


def test_seq_increments_for_same_iteration(tmp_path):
    config = Config(logs_dir=str(tmp_path))
    llm_module._call_seq.clear()

    _write_conversation_log(config, _make_response(), "s", "u", "researcher", 7)
    _write_conversation_log(config, _make_response(), "s", "u", "researcher", 7)
    _write_conversation_log(config, _make_response(), "s", "u", "critic", 7)

    names = sorted(f.name for f in tmp_path.iterdir())
    assert names == [
        "iter007_01_researcher.md",
        "iter007_02_researcher.md",
        "iter007_03_critic.md",
    ]


def test_no_file_when_logs_dir_empty(tmp_path):
    config = Config(logs_dir="")
    llm_module._call_seq.clear()

    _write_conversation_log(config, _make_response(), "s", "u", "researcher", 1)

    # Should not have written anywhere — just verify no crash
    assert True


def test_workspace_init_creates_logs_dir(tmp_path):
    config = Config(workspace_dir=str(tmp_path / "ws"))
    ws = WorkspaceManager(config)
    ws.init("Test problem")

    assert ws.logs_dir.exists()
    assert ws.logs_dir.is_dir()
    assert ws.logs_dir == tmp_path / "ws" / "logs"


# ---------------------------------------------------------------------------
# Tests for _write_agent_conversation_log (tool-use agent logs)
# ---------------------------------------------------------------------------


def _make_agent_result(**overrides):
    defaults = dict(
        text="final text",
        tool_calls=[],
        total_input_tokens=1000,
        total_output_tokens=500,
        rounds=3,
        truncated=False,
        duration=10.0,
        stop_reason="end_turn",
        total_reasoning_tokens=0,
        total_answer_tokens=0,
    )
    defaults.update(overrides)
    return AgentResult(**defaults)


def test_agent_log_single_file_multi_round(tmp_path):
    """Single file created (not N files) for a multi-round session."""
    config = Config(logs_dir=str(tmp_path))
    llm_module._call_seq.clear()

    round_log = [
        {
            "kind": "llm_response",
            "round": 1,
            "text": "round 1 text",
            "tool_calls": [
                {"id": "tc1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
            "input_tokens": 100,
            "output_tokens": 50,
            "reasoning_tokens": 0,
            "answer_tokens": 50,
            "stop_reason": "tool_use",
            "duration": 1.0,
        },
        {
            "kind": "tool_result",
            "round": 1,
            "tool_name": "execute_python",
            "tool_input": {"code": "print(1)"},
            "output": "1\n",
            "is_error": False,
            "duration": 0.5,
        },
        {
            "kind": "llm_response",
            "round": 2,
            "text": "round 2 text",
            "tool_calls": None,
            "input_tokens": 200,
            "output_tokens": 100,
            "reasoning_tokens": 0,
            "answer_tokens": 100,
            "stop_reason": "end_turn",
            "duration": 2.0,
        },
    ]

    result = _make_agent_result(rounds=2)
    _write_agent_conversation_log(
        config, "sys", "user", "computationalist", 1, round_log, result
    )

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "iter001_01_computationalist.md"


def test_agent_log_contains_tool_call_code(tmp_path):
    """File contains tool call name and code in a python block."""
    config = Config(logs_dir=str(tmp_path))
    llm_module._call_seq.clear()

    round_log = [
        {
            "kind": "llm_response",
            "round": 1,
            "text": "",
            "tool_calls": [
                {
                    "id": "tc1",
                    "name": "execute_python",
                    "input": {"code": "import numpy as np\nprint(np.pi)"},
                }
            ],
            "input_tokens": 100,
            "output_tokens": 50,
            "reasoning_tokens": 0,
            "answer_tokens": 0,
            "stop_reason": "tool_use",
            "duration": 1.0,
        },
    ]

    result = _make_agent_result(rounds=1)
    _write_agent_conversation_log(
        config, "sys", "user", "computationalist", 1, round_log, result
    )

    content = (tmp_path / "iter001_01_computationalist.md").read_text()
    assert '<TOOL_CALL name="execute_python">' in content
    assert "~~~python" in content
    assert "import numpy as np" in content


def test_agent_log_contains_tool_result(tmp_path):
    """File contains tool result output and error status."""
    config = Config(logs_dir=str(tmp_path))
    llm_module._call_seq.clear()

    round_log = [
        {
            "kind": "llm_response",
            "round": 1,
            "text": "",
            "tool_calls": [
                {"id": "tc1", "name": "execute_python", "input": {"code": "x"}}
            ],
            "input_tokens": 100,
            "output_tokens": 50,
            "reasoning_tokens": 0,
            "answer_tokens": 0,
            "stop_reason": "tool_use",
            "duration": 1.0,
        },
        {
            "kind": "tool_result",
            "round": 1,
            "tool_name": "execute_python",
            "tool_input": {"code": "x"},
            "output": "NameError: name 'x' is not defined",
            "is_error": True,
            "duration": 0.3,
        },
    ]

    result = _make_agent_result(rounds=1)
    _write_agent_conversation_log(
        config, "sys", "user", "computationalist", 1, round_log, result
    )

    content = (tmp_path / "iter001_01_computationalist.md").read_text()
    assert (
        '<TOOL_RESULT name="execute_python" duration="0.3s" status="error">' in content
    )
    assert "NameError: name 'x' is not defined" in content


def test_agent_log_contains_scaffold_labels(tmp_path):
    """File contains scaffold injection labels."""
    config = Config(logs_dir=str(tmp_path))
    llm_module._call_seq.clear()

    round_log = [
        {
            "kind": "llm_response",
            "round": 1,
            "text": "text",
            "tool_calls": None,
            "input_tokens": 100,
            "output_tokens": 50,
            "reasoning_tokens": 0,
            "answer_tokens": 50,
            "stop_reason": "tool_use",
            "duration": 1.0,
        },
        {
            "kind": "scaffold_injection",
            "round": 1,
            "label": "checkpoint_nudge",
            "content": "CHECKPOINT: You are running low...",
        },
    ]

    result = _make_agent_result(rounds=1)
    _write_agent_conversation_log(
        config, "sys", "user", "computationalist", 1, round_log, result
    )

    content = (tmp_path / "iter001_01_computationalist.md").read_text()
    assert '<USER_MESSAGE label="scaffold: checkpoint_nudge">' in content
    assert "CHECKPOINT: You are running low..." in content


def test_agent_log_xml_tags(tmp_path):
    """XML tags wrap system prompt and user content."""
    config = Config(logs_dir=str(tmp_path))
    llm_module._call_seq.clear()

    round_log = [
        {
            "kind": "llm_response",
            "round": 1,
            "text": "done",
            "tool_calls": None,
            "input_tokens": 100,
            "output_tokens": 50,
            "reasoning_tokens": 0,
            "answer_tokens": 50,
            "stop_reason": "end_turn",
            "duration": 1.0,
        },
    ]

    result = _make_agent_result(rounds=1)
    _write_agent_conversation_log(
        config,
        "system prompt text",
        "user content text",
        "computationalist",
        1,
        round_log,
        result,
    )

    content = (tmp_path / "iter001_01_computationalist.md").read_text()
    assert "<SYSTEM_PROMPT" in content
    assert "system prompt text" in content
    assert "</SYSTEM_PROMPT>" in content
    assert "<USER_MESSAGE" in content
    assert "user content text" in content
    assert "</USER_MESSAGE>" in content


def test_agent_log_seq_increments_by_one(tmp_path):
    """_call_seq increments by 1 (not N) for agent invocations."""
    config = Config(logs_dir=str(tmp_path))
    llm_module._call_seq.clear()

    round_log = [
        {
            "kind": "llm_response",
            "round": 1,
            "text": "r1",
            "tool_calls": None,
            "input_tokens": 100,
            "output_tokens": 50,
            "reasoning_tokens": 0,
            "answer_tokens": 50,
            "stop_reason": "tool_use",
            "duration": 1.0,
        },
        {
            "kind": "llm_response",
            "round": 2,
            "text": "r2",
            "tool_calls": None,
            "input_tokens": 200,
            "output_tokens": 100,
            "reasoning_tokens": 0,
            "answer_tokens": 100,
            "stop_reason": "end_turn",
            "duration": 2.0,
        },
    ]

    result = _make_agent_result(rounds=2)
    _write_agent_conversation_log(
        config, "sys", "user", "computationalist", 1, round_log, result
    )
    _write_agent_conversation_log(
        config, "sys", "user", "computationalist", 1, round_log, result
    )

    files = sorted(f.name for f in tmp_path.iterdir())
    assert files == [
        "iter001_01_computationalist.md",
        "iter001_02_computationalist.md",
    ]
    # seq should be 2 (not 4 which would happen if per-round logging incremented)
    assert llm_module._call_seq[1] == 2


def test_agent_log_no_file_when_logs_dir_empty():
    """No file when logs_dir is empty."""
    config = Config(logs_dir="")
    llm_module._call_seq.clear()

    round_log = [
        {
            "kind": "llm_response",
            "round": 1,
            "text": "text",
            "tool_calls": None,
            "input_tokens": 100,
            "output_tokens": 50,
            "reasoning_tokens": 0,
            "answer_tokens": 50,
            "stop_reason": "end_turn",
            "duration": 1.0,
        },
    ]

    result = _make_agent_result(rounds=1)
    _write_agent_conversation_log(
        config, "sys", "user", "computationalist", 1, round_log, result
    )
    # Should not crash
    assert True


def test_agent_log_forced_final_call(tmp_path):
    """Forced final call section appears with reason label."""
    config = Config(logs_dir=str(tmp_path))
    llm_module._call_seq.clear()

    round_log = [
        {
            "kind": "llm_response",
            "round": 1,
            "text": "",
            "tool_calls": [
                {"id": "tc1", "name": "execute_python", "input": {"code": "pass"}}
            ],
            "input_tokens": 100,
            "output_tokens": 50,
            "reasoning_tokens": 0,
            "answer_tokens": 0,
            "stop_reason": "tool_use",
            "duration": 1.0,
        },
        {
            "kind": "forced_final_call",
            "round": 2,
            "reason": "zero_text",
            "text": "",
            "input_tokens": 500,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "answer_tokens": 0,
            "duration": 1.5,
        },
    ]

    result = _make_agent_result(rounds=2, stop_reason="max_rounds_forced")
    _write_agent_conversation_log(
        config, "sys", "user", "computationalist", 1, round_log, result
    )

    content = (tmp_path / "iter001_01_computationalist.md").read_text()
    assert '<FORCED_FINAL_CALL reason="zero_text"' in content
    assert "*(no text output)*" in content


# ---------------------------------------------------------------------------
# Tests for one-shot log chars/token attributes
# ---------------------------------------------------------------------------


def test_one_shot_log_reasoning_tokens(tmp_path):
    """Reasoning/answer tokens appear when non-zero, absent when zero."""
    config = Config(logs_dir=str(tmp_path))
    llm_module._call_seq.clear()

    # With reasoning tokens
    _write_conversation_log(
        config,
        _make_response(text="x", reasoning_tokens=80, answer_tokens=20),
        "sys",
        "user",
        "planner",
        1,
    )
    content = (tmp_path / "iter001_01_planner.md").read_text()
    assert 'reasoning="80"' in content
    assert 'answer="20"' in content

    # Without reasoning tokens
    _write_conversation_log(
        config, _make_response(text="x"), "sys", "user", "surveyor", 1
    )
    content = (tmp_path / "iter001_02_surveyor.md").read_text()
    assert "reasoning=" not in content


# ---------------------------------------------------------------------------
# Tests for _render_agent_conversation_log
# ---------------------------------------------------------------------------


def test_render_preamble_only():
    """Empty round_log produces system + user sections, no ROUND tags."""
    content = _render_agent_conversation_log("system text", "user text", [])
    assert '<SYSTEM_PROMPT chars="11">' in content
    assert "system text" in content
    assert '<USER_MESSAGE chars="9">' in content
    assert "user text" in content
    assert "<ROUND" not in content


def test_render_with_tools():
    """Tools block appears when tools are provided."""
    tools = [{"function": {"name": "execute_python", "description": "Run code"}}]
    content = _render_agent_conversation_log("sys", "user", [], tools=tools)
    assert "<TOOLS>" in content
    assert "execute_python" in content


def test_render_accumulates_rounds():
    """Rendering with 1 vs 2 rounds produces incremental content."""
    round1 = [
        {
            "kind": "llm_response",
            "round": 1,
            "text": "round1",
            "tool_calls": None,
            "input_tokens": 100,
            "output_tokens": 50,
            "reasoning_tokens": 0,
            "answer_tokens": 50,
            "stop_reason": "end_turn",
            "duration": 1.0,
        },
    ]
    content1 = _render_agent_conversation_log("sys", "user", round1)
    assert '<ROUND n="1">' in content1
    assert '<ROUND n="2">' not in content1

    round2 = round1 + [
        {
            "kind": "llm_response",
            "round": 2,
            "text": "round2",
            "tool_calls": None,
            "input_tokens": 200,
            "output_tokens": 100,
            "reasoning_tokens": 0,
            "answer_tokens": 100,
            "stop_reason": "end_turn",
            "duration": 2.0,
        },
    ]
    content2 = _render_agent_conversation_log("sys", "user", round2)
    assert '<ROUND n="1">' in content2
    assert '<ROUND n="2">' in content2
    assert "round2" in content2
