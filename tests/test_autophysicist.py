"""Tests for the autophysicist mode."""

import textwrap


from open_dirac.autophysicist.memory import PermanentMemory, Scratchpad
from open_dirac.autophysicist.subagent import (
    SubAgentResult,
    _extract_python_code,
)
from open_dirac.autophysicist.tools import ManagerToolExecutor
from open_dirac.config import Config


# ---------------------------------------------------------------------------
# memory.py
# ---------------------------------------------------------------------------


class TestPermanentMemory:
    def test_creates_file(self, tmp_path):
        mem = PermanentMemory(tmp_path)
        assert (tmp_path / "PERMANENT_MEMORY.md").exists()
        assert "# Permanent Memory" in mem.read_full()

    def test_append_and_read(self, tmp_path):
        mem = PermanentMemory(tmp_path)
        msg = mem.append("Result: E = mc^2", iteration=1)
        assert "iteration 1" in msg
        full = mem.read_full()
        assert "## Iteration 1" in full
        assert "E = mc^2" in full

    def test_multiple_appends(self, tmp_path):
        mem = PermanentMemory(tmp_path)
        mem.append("First result", iteration=1)
        mem.append("Second result", iteration=3)
        full = mem.read_full()
        assert "## Iteration 1" in full
        assert "## Iteration 3" in full
        assert full.index("First result") < full.index("Second result")

    def test_size_chars(self, tmp_path):
        mem = PermanentMemory(tmp_path)
        initial_size = mem.size_chars
        mem.append("Some content", iteration=1)
        assert mem.size_chars > initial_size

    def test_reattach_preserves_content(self, tmp_path):
        mem1 = PermanentMemory(tmp_path)
        mem1.append("Persisted result", iteration=5)
        mem2 = PermanentMemory(tmp_path)
        assert "Persisted result" in mem2.read_full()


class TestScratchpad:
    def test_creates_file(self, tmp_path):
        sp = Scratchpad(tmp_path)
        assert (tmp_path / "SCRATCHPAD.md").exists()

    def test_append_and_window(self, tmp_path):
        sp = Scratchpad(tmp_path, window_size=2)
        sp.append("Note A", iteration=1)
        sp.append("Note B", iteration=2)
        sp.append("Note C", iteration=3)
        window = sp.read_window()
        # Only last 2 entries should be visible
        assert "Note A" not in window
        assert "Note B" in window
        assert "Note C" in window

    def test_entry_count(self, tmp_path):
        sp = Scratchpad(tmp_path)
        assert sp.entry_count == 0
        sp.append("A", iteration=1)
        sp.append("B", iteration=2)
        assert sp.entry_count == 2

    def test_reattach_counts_existing(self, tmp_path):
        sp1 = Scratchpad(tmp_path)
        sp1.append("X", iteration=1)
        sp1.append("Y", iteration=2)
        sp2 = Scratchpad(tmp_path)
        assert sp2.entry_count == 2

    def test_read_full(self, tmp_path):
        sp = Scratchpad(tmp_path, window_size=1)
        sp.append("Old note", iteration=1)
        sp.append("New note", iteration=2)
        full = sp.read_full()
        assert "Old note" in full
        assert "New note" in full

    def test_confirmation_message(self, tmp_path):
        sp = Scratchpad(tmp_path, window_size=3)
        msg = sp.append("A note", iteration=1)
        assert "1 of 1" in msg
        sp.append("B", iteration=2)
        msg = sp.append("C", iteration=3)
        assert "3 of 3" in msg
        msg = sp.append("D", iteration=4)
        assert "3 of 4" in msg


# ---------------------------------------------------------------------------
# subagent.py
# ---------------------------------------------------------------------------


class TestExtractPythonCode:
    def test_basic_extraction(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        assert _extract_python_code(text) == "print('hello')"

    def test_no_language_tag(self):
        text = "Code:\n```\nimport math\nprint(math.pi)\n```"
        assert _extract_python_code(text) == "import math\nprint(math.pi)"

    def test_no_code_block(self):
        text = "Just some text without code."
        assert _extract_python_code(text) == ""

    def test_multiline_code(self):
        text = textwrap.dedent("""\
            Explanation here.
            ```python
            import numpy as np
            x = np.linspace(0, 1, 100)
            print(np.sum(x))
            ```
        """)
        code = _extract_python_code(text)
        assert "import numpy" in code
        assert "print(np.sum(x))" in code


class TestSubAgentResult:
    def test_format_no_code(self):
        r = SubAgentResult(
            reasoning_text="The answer is 42.",
            code="",
            execution_output="",
            execution_status="no_code",
            total_input_tokens=100,
            total_output_tokens=50,
        )
        formatted = r.format_for_manager()
        assert "The answer is 42." in formatted
        assert "## Code" not in formatted

    def test_format_with_code(self):
        r = SubAgentResult(
            reasoning_text="Computing pi.",
            code="print(3.14159)",
            execution_output="3.14159",
            execution_status="success",
            total_input_tokens=200,
            total_output_tokens=100,
        )
        formatted = r.format_for_manager()
        assert "Computing pi." in formatted
        assert "print(3.14159)" in formatted
        assert "3.14159" in formatted
        assert "success" in formatted


# ---------------------------------------------------------------------------
# tools.py
# ---------------------------------------------------------------------------


def _make_executor(tmp_path, token_budget=100_000, tool_call_cap=15):
    """Create a ManagerToolExecutor with mocked dependencies."""
    config = Config()
    mem = PermanentMemory(tmp_path)
    sp = Scratchpad(tmp_path)
    return ManagerToolExecutor(
        config=config,
        permanent_memory=mem,
        scratchpad=sp,
        workspace_root=tmp_path,
        iteration=1,
        token_budget=token_budget,
        tool_call_cap=tool_call_cap,
    )


class TestManagerToolExecutor:
    def test_write_permanent_memory(self, tmp_path):
        ex = _make_executor(tmp_path)
        tc = ex.execute("write_to_permanent_memory", {"content": "E = mc^2"})
        assert not tc.is_error
        assert "iteration 1" in tc.output
        assert "E = mc^2" in ex.permanent_memory.read_full()

    def test_write_scratchpad(self, tmp_path):
        ex = _make_executor(tmp_path)
        tc = ex.execute("write_to_scratchpad", {"content": "Try approach B"})
        assert not tc.is_error
        assert "Try approach B" in ex.scratchpad.read_full()

    def test_end_turn_sets_stop(self, tmp_path):
        ex = _make_executor(tmp_path)
        assert not ex.stop_after_round
        tc = ex.execute("end_turn", {})
        assert not tc.is_error
        assert ex.stop_after_round

    def test_empty_content_rejected(self, tmp_path):
        ex = _make_executor(tmp_path)
        tc = ex.execute("write_to_permanent_memory", {"content": "  "})
        assert tc.is_error
        assert "empty" in tc.output.lower()

    def test_unknown_tool(self, tmp_path):
        ex = _make_executor(tmp_path)
        tc = ex.execute("nonexistent_tool", {})
        assert tc.is_error
        assert "Unknown tool" in tc.output

    def test_active_tools_default(self, tmp_path):
        ex = _make_executor(tmp_path)
        tools = ex.active_tools
        names = {t["function"]["name"] for t in tools}
        assert names == {
            "dispatch_subagent",
            "write_to_permanent_memory",
            "write_to_scratchpad",
            "end_turn",
            "submit_final_answer",
        }

    def test_wind_down_removes_dispatch(self, tmp_path):
        ex = _make_executor(tmp_path)
        ex._wind_down = True
        tools = ex.active_tools
        names = {t["function"]["name"] for t in tools}
        assert "dispatch_subagent" not in names
        assert "end_turn" in names

    def test_dispatch_blocked_during_wind_down(self, tmp_path):
        ex = _make_executor(tmp_path)
        ex._wind_down = True
        tc = ex.execute(
            "dispatch_subagent",
            {
                "system_prompt": "You are a physicist.",
                "user_message": "Derive F=ma.",
            },
        )
        assert tc.is_error
        assert "unavailable" in tc.output.lower()

    def test_exit_tool_names(self, tmp_path):
        ex = _make_executor(tmp_path)
        assert ex.exit_tool_name == "end_turn"
        assert ex.exit_tool_names == frozenset({"end_turn", "submit_final_answer"})


class TestBudgetMechanism:
    def test_end_round_triggers_wind_down(self, tmp_path):
        ex = _make_executor(tmp_path, token_budget=1000)
        # Simulate on_round callback updating tokens
        ex.update_manager_tokens(800, 300)
        # end_round should detect budget exceeded
        msg = ex.end_round()
        assert msg is not None
        assert "BUDGET WARNING" in msg
        assert ex._wind_down

    def test_end_round_no_trigger_under_budget(self, tmp_path):
        ex = _make_executor(tmp_path, token_budget=10_000)
        ex.update_manager_tokens(100, 50)
        msg = ex.end_round()
        assert msg is None
        assert not ex._wind_down

    def test_hard_cap_forces_stop(self, tmp_path):
        ex = _make_executor(tmp_path, token_budget=1000)
        # Exceed hard cap (1.5x budget = 1500)
        ex.update_manager_tokens(1000, 600)
        msg = ex.end_round()
        assert msg is not None
        assert "HARD BUDGET" in msg
        assert ex.stop_after_round

    def test_tool_call_cap(self, tmp_path):
        ex = _make_executor(tmp_path, tool_call_cap=3)
        # Simulate 3 tool calls
        ex._tool_call_count = 3
        msg = ex.end_round()
        assert msg is not None
        assert "TOOL CALL CAP" in msg
        assert ex._wind_down

    def test_subagent_tokens_count_toward_budget(self, tmp_path):
        ex = _make_executor(tmp_path, token_budget=1000)
        ex.update_manager_tokens(300, 200)  # 500 manager tokens
        ex.subagent_input_tokens = 400
        ex.subagent_output_tokens = 200  # 600 subagent tokens = 1100 total
        msg = ex.end_round()
        assert msg is not None
        assert ex._wind_down


class TestContextAssembly:
    """Test the _build_user_content function from runner."""

    def test_empty_state(self, tmp_path):
        from open_dirac.autophysicist.runner import _build_user_content

        mem = PermanentMemory(tmp_path)
        sp = Scratchpad(tmp_path)
        content = _build_user_content("What is 2+2?", "", mem, sp, 1, 10)
        assert "Iteration 1 of 10" in content
        assert "What is 2+2?" in content
        assert "Empty" in content  # both memory and scratchpad empty

    def test_with_answer_template(self, tmp_path):
        from open_dirac.autophysicist.runner import _build_user_content

        mem = PermanentMemory(tmp_path)
        sp = Scratchpad(tmp_path)
        template = "def answer():\n    return 42"
        content = _build_user_content("What is 2+2?", template, mem, sp, 1, 10)
        assert "<answer_template>" in content
        assert "def answer():" in content
        assert "```python" in content

    def test_empty_answer_template_omitted(self, tmp_path):
        from open_dirac.autophysicist.runner import _build_user_content

        mem = PermanentMemory(tmp_path)
        sp = Scratchpad(tmp_path)
        content = _build_user_content("What is 2+2?", "", mem, sp, 1, 10)
        assert "<answer_template>" not in content

    def test_with_populated_state(self, tmp_path):
        from open_dirac.autophysicist.runner import _build_user_content

        mem = PermanentMemory(tmp_path)
        mem.append("Verified: 2+2=4", iteration=1)
        sp = Scratchpad(tmp_path)
        sp.append("Next: verify 3+3", iteration=1)
        content = _build_user_content("What is 2+2?", "", mem, sp, 2, 10)
        assert "Iteration 2 of 10" in content
        assert "2+2=4" in content
        assert "verify 3+3" in content
        assert "Empty" not in content
