"""Tests for ToolExecutor, execute_python tool, and run_agent_loop."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from open_dirac.config import Config
from open_dirac.llm import AgentResult, run_agent_loop
from open_dirac.providers.base import ProviderResponse
from open_dirac.agents.computer.tools import ToolExecutor


def _make_executor(timeout: int = 60) -> ToolExecutor:
    """Create a ToolExecutor with a temp workspace."""
    root = Path(tempfile.mkdtemp())
    return ToolExecutor(workspace_root=root, timeout=timeout)


class TestExecutePython:
    def test_simple_script(self):
        executor = _make_executor()
        tc = executor.execute("execute_python", {"code": "print('hello')"})
        assert "hello" in tc.output
        assert "=== tool_exec_001.py ===" in tc.output
        assert not tc.is_error
        assert tc.tool_name == "execute_python"
        assert tc.duration >= 0

    def test_writes_file(self):
        executor = _make_executor()
        executor.execute("execute_python", {"code": "print(1)"})
        script = executor._computations_dir / "tool_exec_001.py"
        assert script.exists()
        assert "print(1)" in script.read_text()

    def test_error_script(self):
        executor = _make_executor()
        tc = executor.execute("execute_python", {"code": "x = 1 +\n"})
        assert tc.is_error
        assert "SyntaxError" in tc.output or "Error" in tc.output

    def test_timeout(self):
        executor = _make_executor(timeout=1)
        tc = executor.execute("execute_python", {"code": "import time; time.sleep(10)"})
        assert tc.is_error
        assert "TIMEOUT" in tc.output
        assert "Suggestions" in tc.output

    def test_output_truncation(self):
        executor = _make_executor()
        # Generate >10K chars of output
        code = "print('x' * 20000)"
        tc = executor.execute("execute_python", {"code": code})
        assert not tc.is_error
        assert len(tc.output) <= 11_200  # 10K body + truncation message + header
        assert "truncated" in tc.output

    def test_unknown_tool_returns_error(self):
        executor = _make_executor()
        tc = executor.execute("nonexistent_tool", {"code": "print(1)"})
        assert tc.is_error
        assert "Unknown tool" in tc.output

    def test_counter_increments(self):
        executor = _make_executor()
        executor.execute("execute_python", {"code": "print(1)"})
        executor.execute("execute_python", {"code": "print(2)"})
        assert (executor._computations_dir / "tool_exec_001.py").exists()
        assert (executor._computations_dir / "tool_exec_002.py").exists()

    def test_name_error_appends_reminder(self):
        executor = _make_executor()
        tc = executor.execute("execute_python", {"code": "print(undefined_var)"})
        assert tc.is_error
        assert "FRESH Python process" in tc.output
        assert "ALL imports and function definitions" in tc.output

    def test_non_name_error_no_reminder(self):
        executor = _make_executor()
        tc = executor.execute("execute_python", {"code": "1/0"})
        assert tc.is_error
        assert "FRESH Python process" not in tc.output


class TestFilenameHandling:
    def test_sanitize_strips_path_separators(self):
        result = ToolExecutor._sanitize_filename("../../etc/passwd.py")
        assert "/" not in result
        assert "\\" not in result
        assert ".." not in result
        assert result.endswith(".py")

    def test_sanitize_ensures_py_extension(self):
        assert ToolExecutor._sanitize_filename("verify_enum").endswith(".py")
        assert ToolExecutor._sanitize_filename("verify_enum.txt").endswith(".py")

    def test_sanitize_truncates_long_names(self):
        long_name = "a" * 100 + ".py"
        result = ToolExecutor._sanitize_filename(long_name, max_len=60)
        assert len(result) <= 60
        assert result.endswith(".py")

    def test_fallback_naming_without_filename(self):
        executor = _make_executor()
        executor.execute("execute_python", {"code": "print(1)", "purpose": "test"})
        assert (executor._computations_dir / "tool_exec_001.py").exists()

    def test_named_script_with_counter(self):
        executor = _make_executor()
        executor.execute(
            "execute_python",
            {
                "code": "print(1)",
                "purpose": "test",
                "filename": "verify_enum.py",
            },
        )
        assert (executor._computations_dir / "001_verify_enum.py").exists()

    def test_agent_counter_prefix_stripped(self):
        """Agent-provided filenames like '002_verify_exact.py' should not get double-prefixed."""
        executor = _make_executor()
        executor.execute(
            "execute_python",
            {
                "code": "print(1)",
                "purpose": "test",
                "filename": "002_verify_exact.py",
            },
        )
        assert (executor._computations_dir / "001_verify_exact.py").exists()
        assert not (executor._computations_dir / "001_002_verify_exact.py").exists()

    def test_agent_counter_prefix_stripped_various(self):
        """Multiple counter-prefix patterns are handled."""
        executor = _make_executor()
        executor.execute(
            "execute_python",
            {
                "code": "print(1)",
                "purpose": "a",
                "filename": "01_foo.py",
            },
        )
        executor.execute(
            "execute_python",
            {
                "code": "print(2)",
                "purpose": "b",
                "filename": "0003_bar.py",
            },
        )
        assert executor._script_names == ["001_foo.py", "002_bar.py"]

    def test_script_names_tracking(self):
        executor = _make_executor()
        executor.execute("execute_python", {"code": "print(1)", "purpose": "a"})
        executor.execute(
            "execute_python",
            {
                "code": "print(2)",
                "purpose": "b",
                "filename": "second.py",
            },
        )
        assert executor._script_names == ["tool_exec_001.py", "002_second.py"]

    def test_structured_header_in_output(self):
        executor = _make_executor()
        tc = executor.execute(
            "execute_python",
            {
                "code": "print('ok')",
                "purpose": "Check sanity",
                "filename": "sanity.py",
            },
        )
        assert "=== 001_sanity.py ===" in tc.output
        assert "Purpose: Check sanity" in tc.output
        assert "Exit: success" in tc.output
        assert "ok" in tc.output

    def test_output_file_created(self):
        executor = _make_executor()
        executor.execute(
            "execute_python", {"code": "print('hello world')", "purpose": "test"}
        )
        output_file = executor._computations_dir / "tool_exec_001.output"
        assert output_file.exists()
        assert "hello world" in output_file.read_text()

    def test_error_output_file_includes_stderr(self):
        executor = _make_executor()
        executor.execute(
            "execute_python", {"code": "import sys; sys.exit(1)", "purpose": "test"}
        )
        output_file = executor._computations_dir / "tool_exec_001.output"
        assert output_file.exists()


class TestToolDefinitions:
    def test_definitions_format(self):
        defs = ToolExecutor.TOOL_DEFINITIONS
        assert len(defs) == 3
        names = {d["function"]["name"] for d in defs}
        assert names == {"document_approach", "execute_python", "submit_result"}
        for d in defs:
            assert d["type"] == "function"

    def test_execute_python_requires_purpose(self):
        func = next(
            d["function"]
            for d in ToolExecutor.TOOL_DEFINITIONS
            if d["function"]["name"] == "execute_python"
        )
        assert func["name"] == "execute_python"
        props = func["parameters"]["properties"]
        assert "purpose" in props
        assert props["purpose"]["type"] == "string"
        assert set(func["parameters"]["required"]) == {"purpose", "code"}

    def test_execute_python_has_optional_filename(self):
        func = ToolExecutor._EXECUTE_PYTHON_DEF["function"]
        props = func["parameters"]["properties"]
        assert "filename" in props
        assert props["filename"]["type"] == "string"
        assert "filename" not in func["parameters"]["required"]

    def test_document_approach_schema(self):
        func = next(
            d["function"]
            for d in ToolExecutor.TOOL_DEFINITIONS
            if d["function"]["name"] == "document_approach"
        )
        props = func["parameters"]["properties"]
        assert "approach" in props
        assert "assumptions" in props
        assert set(func["parameters"]["required"]) == {"approach"}

    def test_report_progress_schema(self):
        func = ToolExecutor._REPORT_PROGRESS_DEF["function"]
        assert func["name"] == "report_progress"
        props = func["parameters"]["properties"]
        assert set(props.keys()) == {
            "findings_so_far",
            "remaining_questions",
            "ready_to_conclude",
        }
        assert props["ready_to_conclude"]["type"] == "boolean"
        assert set(func["parameters"]["required"]) == {
            "findings_so_far",
            "remaining_questions",
            "ready_to_conclude",
        }

    def test_submit_result_has_evidence_scripts(self):
        func = ToolExecutor._SUBMIT_RESULT_DEF["function"]
        props = func["parameters"]["properties"]
        assert "evidence_scripts" in props
        assert props["evidence_scripts"]["type"] == "array"
        assert "evidence_scripts" not in func["parameters"]["required"]


class TestTruncation:
    def test_short_text_unchanged(self):
        assert ToolExecutor._truncate_output("short") == "short"

    def test_long_text_truncated(self):
        text = "a" * 20_000
        result = ToolExecutor._truncate_output(text, limit=10_000)
        assert len(result) < 11_000
        assert "truncated" in result
        # Preserves head and tail
        assert result.startswith("a" * 100)
        assert result.endswith("a" * 100)


class TestDocumentApproach:
    def test_stores_approach(self):
        executor = _make_executor()
        params = {
            "approach": "Compute partition function via SymPy",
            "assumptions": "T > 0",
        }
        tc = executor.execute("document_approach", params)
        assert not tc.is_error
        assert "documented" in tc.output.lower()
        assert (
            executor._documented_approach["approach"]
            == "Compute partition function via SymPy"
        )

    def test_does_not_stop(self):
        executor = _make_executor()
        tc = executor.execute("document_approach", {"approach": "test"})
        assert (
            not hasattr(executor, "stop_after_round") or not executor.stop_after_round
        )


class TestActiveToolsLifecycle:
    """Test dynamic tool switching: document_approach → execute_python, progress check."""

    def test_initial_tools_before_approach(self):
        """Before approach, only document_approach and submit_result are available."""
        from open_dirac.task import TaskType

        root = Path(tempfile.mkdtemp())
        executor = ToolExecutor(workspace_root=root, task_type=TaskType.COMPUTE)
        tools = executor.active_tools
        names = {t["function"]["name"] for t in tools}
        assert names == {"document_approach", "submit_result"}

    def test_tools_after_approach(self):
        """After document_approach, execute_python replaces it."""
        from open_dirac.task import TaskType

        root = Path(tempfile.mkdtemp())
        executor = ToolExecutor(workspace_root=root, task_type=TaskType.COMPUTE)
        executor.execute("document_approach", {"approach": "test"})
        tools = executor.active_tools
        names = {t["function"]["name"] for t in tools}
        assert names == {"execute_python", "submit_result"}

    def test_progress_check_exposes_report_progress(self):
        """Setting _progress_check_pending adds report_progress to tools."""
        from open_dirac.task import TaskType

        root = Path(tempfile.mkdtemp())
        executor = ToolExecutor(workspace_root=root, task_type=TaskType.COMPUTE)
        executor._approach_documented = True
        executor._progress_check_pending = True
        tools = executor.active_tools
        names = {t["function"]["name"] for t in tools}
        assert "report_progress" in names

    def test_report_progress_clears_pending(self):
        """Calling report_progress removes it from next tool set."""
        from open_dirac.task import TaskType

        root = Path(tempfile.mkdtemp())
        executor = ToolExecutor(workspace_root=root, task_type=TaskType.COMPUTE)
        executor._approach_documented = True
        executor._progress_check_pending = True
        executor.execute(
            "report_progress",
            {
                "findings_so_far": "done",
                "remaining_questions": "",
                "ready_to_conclude": False,
            },
        )
        assert executor._progress_check_pending is False
        names = {t["function"]["name"] for t in executor.active_tools}
        assert "report_progress" not in names

    def test_non_compute_returns_initial(self):
        """Non-COMPUTE task type gets initial tool set (document_approach + submit_result)."""
        from open_dirac.task import TaskType

        root = Path(tempfile.mkdtemp())
        executor = ToolExecutor(workspace_root=root, task_type=TaskType.RESEARCH)
        tools = executor.active_tools
        names = {t["function"]["name"] for t in tools}
        assert names == {"document_approach", "submit_result"}


class TestSubmitResult:
    def test_sets_stop_flag(self):
        executor = _make_executor()
        params = {
            "target_id": "WH-001",
            "description": "Computed F(p)",
            "method": "numerical",
            "result": "F(p) = 0.99",
            "confidence": "approximate",
            "notes": "Convergent.",
        }
        tc = executor.execute("submit_result", params)
        assert not tc.is_error
        assert "WH-001" in tc.output
        assert executor.stop_after_round is True

    def test_stores_last_result(self):
        executor = _make_executor()
        params = {
            "target_id": "WH-002",
            "description": "Computed entropy",
            "method": "analytical",
            "result": "S = k ln(W)",
            "confidence": "exact",
            "notes": "Standard formula.",
        }
        executor.execute("submit_result", params)
        assert executor._last_result == params

    def test_output_message(self):
        executor = _make_executor()
        tc = executor.execute(
            "submit_result",
            {
                "target_id": "WH-003",
                "description": "d",
                "method": "m",
                "result": "r",
                "confidence": "partial",
                "notes": "n",
            },
        )
        assert "WH-003" in tc.output
        assert "partial" in tc.output
        assert tc.tool_name == "submit_result"

    def test_researcher_style_no_target_id(self):
        """Researcher-style submit_result (no target_id, has summary)."""
        executor = _make_executor()
        params = {
            "reasoning": "By direct computation...",
            "result": "T_H = 1/(8*pi*M)",
            "method": "analytical derivation",
            "confidence": "exact",
            "summary": "Hawking temperature derived via Euclidean path integral",
        }
        tc = executor.execute("submit_result", params)
        assert not tc.is_error
        assert executor.stop_after_round is True
        assert "Hawking temperature" in tc.output
        assert "WH-" not in tc.output  # no target_id in output

    def test_researcher_style_summary_truncated(self):
        """Researcher-style output truncates long summaries to 80 chars."""
        executor = _make_executor()
        long_summary = "A" * 100
        tc = executor.execute(
            "submit_result",
            {
                "reasoning": "...",
                "result": "...",
                "method": "m",
                "confidence": "exact",
                "summary": long_summary,
            },
        )
        # The label should be truncated
        assert len(tc.output) < len(long_summary) + 30  # "Result recorded: " + 80 chars


class TestToolSetsForTaskType:
    def test_computer_tools(self):
        from open_dirac.task import TaskType

        tools = ToolExecutor.tools_for_task_type(TaskType.COMPUTE)
        names = {t["function"]["name"] for t in tools}
        assert names == {"document_approach", "execute_python", "submit_result"}


class TestExitToolName:
    """Test exit_tool_name property on all executor types."""

    def test_tool_executor_exit_tool(self):
        root = Path(tempfile.mkdtemp())
        executor = ToolExecutor(workspace_root=root)
        assert executor.exit_tool_name == "submit_result"

    def test_orchestrator_exit_tools(self):
        from open_dirac.agents.orchestrator.tools import OrchestratorToolExecutor

        expected = {
            "add_hypothesis",
            "dispatch_researcher",
            "dispatch_computer",
            "request_termination",
        }
        assert OrchestratorToolExecutor.exit_tool_names == frozenset(expected)

    def test_report_progress_mentions_submit_result(self):
        from open_dirac.task import TaskType

        root = Path(tempfile.mkdtemp())
        executor = ToolExecutor(workspace_root=root, task_type=TaskType.COMPUTE)
        tc = executor.execute(
            "report_progress",
            {
                "findings_so_far": "done",
                "remaining_questions": "",
                "ready_to_conclude": True,
            },
        )
        assert "submit_result" in tc.output


# --- Agent loop tests ---


def _mock_provider_response(
    text="", stop_reason="end_turn", input_tokens=100, output_tokens=50, tool_calls=None
):
    """Create a mock ProviderResponse."""
    return ProviderResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason=stop_reason,
        tool_calls=tool_calls,
        raw_content=None,
    )


class TestReportProgress:
    def test_report_progress_does_not_stop(self):
        executor = _make_executor()
        params = {
            "findings_so_far": "42 found",
            "remaining_questions": "none",
            "ready_to_conclude": True,
        }
        tc = executor.execute("report_progress", params)
        assert not tc.is_error
        assert not hasattr(executor, "stop_after_round") or not getattr(
            executor, "stop_after_round", False
        )

    def test_report_progress_ready_message(self):
        executor = _make_executor()
        tc = executor.execute(
            "report_progress",
            {
                "findings_so_far": "done",
                "remaining_questions": "",
                "ready_to_conclude": True,
            },
        )
        assert "submit_result" in tc.output

    def test_report_progress_not_ready_message(self):
        executor = _make_executor()
        tc = executor.execute(
            "report_progress",
            {
                "findings_so_far": "partial",
                "remaining_questions": "need more data",
                "ready_to_conclude": False,
            },
        )
        assert "Continue" in tc.output
        assert "need more data" in tc.output


def _make_config(**overrides) -> Config:
    defaults = dict(
        api_key="test-key",
        logs_dir="",
        provider="anthropic",
        progress_check_interval=999,
    )  # disable progress checks by default in tests
    defaults.update(overrides)
    return Config(**defaults)


def _mock_provider():
    """Create a mock provider with sensible defaults for format methods."""
    provider = MagicMock()
    provider.format_assistant_message.return_value = {
        "role": "assistant",
        "content": "mock",
    }
    provider.build_tool_result_messages.return_value = [{"role": "user", "content": []}]
    provider.prepare_messages.side_effect = lambda msgs: msgs
    return provider


class TestAgentLoop:
    @patch("open_dirac.llm._get_provider")
    def test_end_turn_first_call(self, mock_get_provider):
        """LLM returns text with end_turn on the first call."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider
        provider.call.return_value = _mock_provider_response(
            "Final answer.", "end_turn", 200, 100
        )

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=5,
        )

        assert isinstance(result, AgentResult)
        assert result.text == "Final answer."
        assert result.rounds == 1
        assert result.tool_calls == []
        assert not result.truncated
        assert result.stop_reason == "end_turn"

    @patch("open_dirac.llm._get_provider")
    def test_one_tool_call_then_end(self, mock_get_provider):
        """LLM calls a tool in round 1, then returns text in round 2 → text_end_turn_recovery → submit_result."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: tool_use
        round1 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(42)"}}
            ],
        )
        # Round 2: end_turn with text → text_end_turn_recovery fires
        round2 = _mock_provider_response(
            "Done. VERDICT: VERIFIED", "end_turn", 300, 100
        )
        # Round 3: after recovery, model calls submit_result
        round3 = _mock_provider_response(
            "",
            "tool_use",
            150,
            60,
            tool_calls=[
                {
                    "id": "t2",
                    "name": "submit_result",
                    "input": {
                        "target_id": "RQ-001",
                        "description": "verified",
                        "method": "numerical",
                        "result": "ok",
                        "confidence": "exact",
                        "notes": "done",
                    },
                }
            ],
        )
        provider.call.side_effect = [round1, round2, round3]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=5,
        )

        assert result.rounds == 3
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "execute_python"
        assert result.stop_reason == "executor_stop"
        assert not result.truncated

    @patch("open_dirac.llm._get_provider")
    def test_max_rounds_exhausted(self, mock_get_provider):
        """LLM always returns tool_use — stops at max_rounds with forced final call."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        tool_response = _mock_provider_response(
            "",
            "tool_use",
            100,
            50,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
        )
        text_response = _mock_provider_response(
            "## COMP-001\n**VERDICT:** INCONCLUSIVE", "end_turn", 150, 80
        )
        provider.call.side_effect = [
            tool_response,
            tool_response,
            tool_response,  # 3 tool-use rounds
            text_response,
            text_response,
            text_response,  # 3 forced final retries
        ]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=3,
        )

        assert result.rounds == 6  # 3 tool rounds + 3 forced retries
        assert result.truncated
        assert result.stop_reason == "max_rounds_forced"
        assert len(result.tool_calls) == 3
        assert provider.call.call_count == 6

    @patch("open_dirac.llm._get_provider")
    def test_token_accumulation(self, mock_get_provider):
        """Tokens are summed across rounds."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        round1 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
        )
        round2 = _mock_provider_response(
            "",
            "tool_use",
            300,
            90,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
        )
        # Round 3: text end_turn → text_end_turn_recovery fires
        round3 = _mock_provider_response("Done.", "end_turn", 400, 100)
        # Round 4: after recovery, model calls submit_result
        round4 = _mock_provider_response(
            "",
            "tool_use",
            150,
            60,
            tool_calls=[
                {
                    "id": "t2",
                    "name": "submit_result",
                    "input": {
                        "target_id": "RQ-001",
                        "description": "done",
                        "method": "numerical",
                        "result": "ok",
                        "confidence": "exact",
                        "notes": "done",
                    },
                }
            ],
        )
        provider.call.side_effect = [round1, round2, round3, round4]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=5,
        )

        assert result.total_input_tokens == 1050  # 200+300+400+150
        assert result.total_output_tokens == 330  # 80+90+100+60
        assert result.rounds == 4

    @patch("open_dirac.llm._get_provider")
    def test_max_tokens_stop(self, mock_get_provider):
        """stop_reason=max_tokens in round 1 returns truncated."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider
        provider.call.return_value = _mock_provider_response(
            "Partial...", "max_tokens", 200, 16384
        )

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=5,
        )

        assert result.truncated
        assert result.stop_reason == "max_tokens"
        assert result.rounds == 1


class TestForcedPartialOutput:
    """Test the forced text-only final call when max_rounds is exhausted."""

    @patch("open_dirac.llm._get_provider")
    def test_forced_call_includes_exit_tool(self, mock_get_provider):
        """Forced final call should include the exit tool (submit_result)."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        tool_response = _mock_provider_response(
            "",
            "tool_use",
            100,
            50,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
        )
        text_response = _mock_provider_response(
            "## COMP-001\n**VERDICT:** INCONCLUSIVE", "end_turn", 150, 80
        )
        provider.call.side_effect = [
            tool_response,
            tool_response,  # 2 rounds of tool use
            text_response,
            text_response,
            text_response,  # 3 forced final retries
        ]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=2,
        )

        calls = provider.call.call_args_list
        assert len(calls) == 5  # 2 tool rounds + 3 forced retries
        # First two calls should have full tool set
        assert calls[0].kwargs.get("tools") is not None
        # Forced calls should include only the exit tool
        forced_tools = calls[2].kwargs.get("tools")
        assert forced_tools is not None
        tool_names = {t["function"]["name"] for t in forced_tools}
        assert tool_names == {"submit_result"}
        # System prompt is unchanged (no mutation)
        assert calls[2].kwargs.get("system") == "sys"

    @patch("open_dirac.llm._get_provider")
    def test_forced_call_text_in_result(self, mock_get_provider):
        """Result text should come from the forced final call."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        tool_response = _mock_provider_response(
            "",
            "tool_use",
            100,
            50,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
        )
        text_response = _mock_provider_response(
            "Forced partial output here", "end_turn", 150, 80
        )
        provider.call.side_effect = [
            tool_response,
            text_response,
            text_response,
            text_response,
        ]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=1,
        )

        assert result.text == "Forced partial output here"
        assert result.stop_reason == "max_rounds_forced"

    @patch("open_dirac.llm._get_provider")
    def test_token_accumulation_includes_forced(self, mock_get_provider):
        """Total tokens should include all forced call retries."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        tool_response = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
        )
        text_response = _mock_provider_response("Final", "end_turn", 300, 120)
        provider.call.side_effect = [
            tool_response,
            text_response,
            text_response,
            text_response,
        ]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=1,
        )

        assert result.total_input_tokens == 1100  # 200 + 3*300
        assert result.total_output_tokens == 440  # 80 + 3*120
        assert result.rounds == 4  # 1 tool round + 3 forced retries

    @patch("open_dirac.llm._get_provider")
    def test_stop_reason_is_max_rounds_forced(self, mock_get_provider):
        """Stop reason should be 'max_rounds_forced'."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        tool_response = _mock_provider_response(
            "",
            "tool_use",
            100,
            50,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
        )
        text_response = _mock_provider_response("Done", "end_turn", 100, 50)
        provider.call.side_effect = [
            tool_response,
            text_response,
            text_response,
            text_response,
        ]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=1,
        )

        assert result.stop_reason == "max_rounds_forced"
        assert result.truncated is True


class TestEmptyTextFallthrough:
    """Test Gap A: end_turn with empty text falls through to forced final call."""

    @patch("open_dirac.llm._get_provider")
    def test_empty_end_turn_recovery_then_text(self, mock_get_provider):
        """First empty end_turn injects recovery; model responds with text → text_end_turn_recovery → falls through."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: tool_use with code execution
        round1 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(42)"}}
            ],
        )
        # Round 2: end_turn but empty text -> empty_end_turn_recovery injected
        round2 = _mock_provider_response("", "end_turn", 150, 0)
        # Round 3: model responds with text → text_end_turn_recovery fires (count=1)
        round3 = _mock_provider_response(
            "## COMP-001: Result\n**VERDICT:** INCONCLUSIVE", "end_turn", 300, 100
        )
        # Round 4: model responds with text again → falls through (count=2)
        round4 = _mock_provider_response(
            "## COMP-001: Result\n**VERDICT:** INCONCLUSIVE", "end_turn", 300, 100
        )
        provider.call.side_effect = [round1, round2, round3, round4]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=6,
        )

        assert "COMP-001" in result.text
        assert result.stop_reason == "end_turn"
        assert provider.call.call_count == 4

    @patch("open_dirac.llm._get_provider")
    def test_empty_end_turns_retry_until_max_rounds(self, mock_get_provider):
        """Empty end_turns retry until max_rounds, then forced final with exit tool."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: tool_use
        round1 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(42)"}}
            ],
        )
        # Rounds 2-3: empty end_turns -> recovery injected each time
        round2 = _mock_provider_response("", "end_turn", 150, 0)
        round3 = _mock_provider_response("", "end_turn", 150, 0)
        # Forced final calls: model returns text (no exit tool called)
        forced1 = _mock_provider_response("Partial.", "end_turn", 150, 60)
        forced2 = _mock_provider_response("More.", "end_turn", 150, 60)
        forced3 = _mock_provider_response(
            "## COMP-001: Forced result", "end_turn", 300, 100
        )
        provider.call.side_effect = [round1, round2, round3, forced1, forced2, forced3]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=3,
        )

        assert "COMP-001" in result.text
        assert result.stop_reason == "max_rounds_forced"
        # 3 main (1 tool_use + 2 empty) + 3 forced retries
        assert provider.call.call_count == 6
        # Forced calls included exit tool
        forced_call = provider.call.call_args_list[3]
        assert forced_call.kwargs.get("tools") is not None

    @patch("open_dirac.llm._get_provider")
    def test_empty_end_turn_with_text_triggers_recovery(self, mock_get_provider):
        """end_turn with text after tool calls triggers text_end_turn_recovery."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: tool_use
        round1 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
        )
        # Round 2: end_turn with real text → text_end_turn_recovery fires
        round2 = _mock_provider_response(
            "## COMP-001\n**VERDICT:** VERIFIED", "end_turn", 300, 100
        )
        # Round 3: after recovery, model calls submit_result
        round3 = _mock_provider_response(
            "",
            "tool_use",
            150,
            60,
            tool_calls=[
                {
                    "id": "t2",
                    "name": "submit_result",
                    "input": {
                        "target_id": "RQ-001",
                        "description": "verified",
                        "method": "numerical",
                        "result": "ok",
                        "confidence": "exact",
                        "notes": "done",
                    },
                }
            ],
        )
        provider.call.side_effect = [round1, round2, round3]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=5,
        )

        assert result.stop_reason == "executor_stop"
        assert provider.call.call_count == 3


class TestProgressCheckInLoop:
    """Test progress check injection after consecutive execute_python calls."""

    @patch("open_dirac.llm._get_provider")
    def test_progress_check_injected_after_n_rounds(self, mock_get_provider):
        """Progress check message injected after progress_check_interval consecutive exec_python."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        exec_resp = _mock_provider_response(
            "",
            "tool_use",
            100,
            50,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
        )
        # After 3 exec_python rounds, progress check is injected;
        # model calls report_progress then submit_result
        progress_resp = _mock_provider_response(
            "",
            "tool_use",
            100,
            50,
            tool_calls=[
                {
                    "id": "t2",
                    "name": "report_progress",
                    "input": {
                        "findings_so_far": "42",
                        "remaining_questions": "",
                        "ready_to_conclude": True,
                    },
                }
            ],
        )
        result_resp = _mock_provider_response(
            "",
            "tool_use",
            100,
            50,
            tool_calls=[
                {
                    "id": "t3",
                    "name": "submit_result",
                    "input": {
                        "target_id": "RQ-001",
                        "description": "computed",
                        "method": "num",
                        "result": "ok",
                        "confidence": "exact",
                        "notes": "done",
                    },
                }
            ],
        )
        provider.call.side_effect = [
            exec_resp,
            exec_resp,
            exec_resp,
            progress_resp,
            result_resp,
        ]

        config = _make_config(progress_check_interval=3)
        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="q",
            config=config,
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=10,
        )

        assert result.stop_reason == "executor_stop"
        # Check progress check message was injected in round 4's messages
        calls = provider.call.call_args_list
        round4_messages = calls[3].kwargs["messages"]
        progress_found = any(
            isinstance(msg.get("content"), str) and "PROGRESS CHECK" in msg["content"]
            for msg in round4_messages
            if isinstance(msg, dict) and msg.get("role") == "user"
        )
        assert progress_found, (
            "Progress check message should be injected after 3 exec_python rounds"
        )

    @patch("open_dirac.llm._get_provider")
    def test_no_progress_check_before_interval(self, mock_get_provider):
        """No progress check if fewer than N exec_python rounds."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        exec_resp = _mock_provider_response(
            "",
            "tool_use",
            100,
            50,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
        )
        final_resp = _mock_provider_response("Done.", "end_turn", 100, 50)
        submit_resp = _mock_provider_response(
            "",
            "tool_use",
            100,
            50,
            tool_calls=[
                {
                    "id": "t2",
                    "name": "submit_result",
                    "input": {
                        "target_id": "RQ-001",
                        "description": "done",
                        "method": "numerical",
                        "result": "ok",
                        "confidence": "exact",
                        "notes": "done",
                    },
                }
            ],
        )
        # 2 exec → text end_turn → recovery → submit_result
        provider.call.side_effect = [exec_resp, exec_resp, final_resp, submit_resp]

        config = _make_config(progress_check_interval=3)
        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="q",
            config=config,
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=10,
        )

        assert result.stop_reason == "executor_stop"
        # No PROGRESS CHECK in any messages
        for call in provider.call.call_args_list:
            msgs = call.kwargs.get("messages", [])
            for msg in msgs:
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    assert "PROGRESS CHECK" not in msg["content"]

    @patch("open_dirac.llm._get_provider")
    def test_progress_check_resets_after_report_progress(self, mock_get_provider):
        """Counter resets after report_progress; no injection at next interval unless earned."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        exec_resp = _mock_provider_response(
            "",
            "tool_use",
            100,
            50,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
        )
        progress_resp = _mock_provider_response(
            "",
            "tool_use",
            100,
            50,
            tool_calls=[
                {
                    "id": "t2",
                    "name": "report_progress",
                    "input": {
                        "findings_so_far": "partial",
                        "remaining_questions": "more",
                        "ready_to_conclude": False,
                    },
                }
            ],
        )
        final_resp = _mock_provider_response("Done.", "end_turn", 100, 50)
        submit_resp = _mock_provider_response(
            "",
            "tool_use",
            100,
            50,
            tool_calls=[
                {
                    "id": "t3",
                    "name": "submit_result",
                    "input": {
                        "target_id": "RQ-001",
                        "description": "done",
                        "method": "numerical",
                        "result": "ok",
                        "confidence": "exact",
                        "notes": "done",
                    },
                }
            ],
        )
        # 2 exec → progress → 1 exec → text end_turn → recovery → submit_result
        provider.call.side_effect = [
            exec_resp,
            exec_resp,
            progress_resp,
            exec_resp,
            final_resp,
            submit_resp,
        ]

        config = _make_config(progress_check_interval=2)
        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="q",
            config=config,
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=10,
        )

        assert result.stop_reason == "executor_stop"
        # Progress check should have fired once (after round 2), not after round 4.
        # Count PROGRESS CHECK messages in the LAST call's messages (they accumulate).
        last_messages = provider.call.call_args_list[-1].kwargs["messages"]
        progress_count = sum(
            1
            for msg in last_messages
            if isinstance(msg, dict)
            and isinstance(msg.get("content"), str)
            and "PROGRESS CHECK" in msg["content"]
        )
        assert progress_count == 1


class TestSubmitResultInLoop:
    """Test submit_result triggers executor_stop in agent loop."""

    @patch("open_dirac.llm._get_provider")
    def test_submit_result_stops_loop(self, mock_get_provider):
        """Round 1: execute_python, round 2: submit_result → executor_stop."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: tool_use with execute_python
        round1 = _mock_provider_response(
            "Computing...",
            "tool_use",
            200,
            80,
            tool_calls=[
                {
                    "id": "t1",
                    "name": "execute_python",
                    "input": {"purpose": "Check result", "code": "print(1.0)"},
                }
            ],
        )
        # Round 2: tool_use with submit_result
        round2 = _mock_provider_response(
            "",
            "tool_use",
            150,
            60,
            tool_calls=[
                {
                    "id": "t2",
                    "name": "submit_result",
                    "input": {
                        "target_id": "RQ-001",
                        "description": "computed value",
                        "method": "numerical",
                        "result": "value=1.0",
                        "confidence": "exact",
                        "notes": "Done.",
                    },
                }
            ],
        )
        provider.call.side_effect = [round1, round2]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=10,
        )

        assert result.stop_reason == "executor_stop"
        assert result.rounds == 2
        assert len(result.tool_calls) == 2
        assert result.tool_calls[1].tool_name == "submit_result"
        assert not result.truncated
        assert provider.call.call_count == 2


class TestReadyToConcludeRecovery:
    """Test ready-to-conclude recovery re-prompt (Part A)."""

    @patch("open_dirac.llm._get_provider")
    def test_ready_conclude_recovery_then_exit_tool(self, mock_get_provider):
        """report_progress(ready=True) → end_turn with text → recovery → submit_result → executor_stop."""
        from open_dirac.task import TaskType

        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: report_progress with ready_to_conclude=True
        round1 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {
                    "id": "t1",
                    "name": "report_progress",
                    "input": {
                        "findings_so_far": "Found the answer",
                        "remaining_questions": "",
                        "ready_to_conclude": True,
                    },
                }
            ],
        )
        # Round 2: end_turn with text (model wrote answer as markdown instead of calling exit tool)
        round2 = _mock_provider_response(
            "The result is S = k ln(W).", "end_turn", 150, 60
        )
        # Round 3: after recovery re-prompt, model calls submit_result
        round3 = _mock_provider_response(
            "",
            "tool_use",
            150,
            60,
            tool_calls=[
                {
                    "id": "t2",
                    "name": "submit_result",
                    "input": {
                        "target_id": "WH-001",
                        "description": "Entropy formula",
                        "method": "analytical",
                        "result": "S = k ln(W)",
                        "confidence": "exact",
                        "notes": "Standard result.",
                    },
                }
            ],
        )
        provider.call.side_effect = [round1, round2, round3]

        root = Path(tempfile.mkdtemp())
        executor = ToolExecutor(workspace_root=root, task_type=TaskType.COMPUTE)
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.COMPUTER_TOOLS,
            max_rounds=5,
        )

        assert result.stop_reason == "executor_stop"
        assert result.rounds == 3
        # Recovery message was injected
        calls = provider.call.call_args_list
        round3_messages = calls[2].kwargs["messages"]
        recovery_found = any(
            isinstance(msg.get("content"), str)
            and "ready to conclude" in msg["content"]
            for msg in round3_messages
            if isinstance(msg, dict) and msg.get("role") == "user"
        )
        assert recovery_found, "Recovery re-prompt should mention ready to conclude"

    @patch("open_dirac.llm._get_provider")
    def test_ready_conclude_no_flag_triggers_text_recovery(self, mock_get_provider):
        """end_turn with text but no ready_to_conclude → text_end_turn_recovery fires."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: report_progress with ready_to_conclude=False
        round1 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {
                    "id": "t1",
                    "name": "report_progress",
                    "input": {
                        "findings_so_far": "Still working",
                        "remaining_questions": "need more data",
                        "ready_to_conclude": False,
                    },
                }
            ],
        )
        # Round 2: end_turn with text → text_end_turn_recovery fires
        round2 = _mock_provider_response("Here is my conclusion.", "end_turn", 150, 60)
        # Round 3: after recovery, model calls submit_result
        round3 = _mock_provider_response(
            "",
            "tool_use",
            150,
            60,
            tool_calls=[
                {
                    "id": "t2",
                    "name": "submit_result",
                    "input": {
                        "target_id": "RQ-001",
                        "description": "conclusion",
                        "method": "numerical",
                        "result": "ok",
                        "confidence": "exact",
                        "notes": "done",
                    },
                }
            ],
        )
        provider.call.side_effect = [round1, round2, round3]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=5,
        )

        assert result.stop_reason == "executor_stop"
        assert result.rounds == 3
        assert provider.call.call_count == 3

    @patch("open_dirac.llm._get_provider")
    def test_ready_conclude_recovery_second_end_turn_falls_through(
        self, mock_get_provider
    ):
        """Recovery once → model again end_turn with text → falls through to normal end_turn."""
        from open_dirac.task import TaskType

        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: report_progress with ready_to_conclude=True
        round1 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {
                    "id": "t1",
                    "name": "report_progress",
                    "input": {
                        "findings_so_far": "Found answer",
                        "remaining_questions": "",
                        "ready_to_conclude": True,
                    },
                }
            ],
        )
        # Round 2: end_turn with text → recovery injected
        round2 = _mock_provider_response("The answer is 42.", "end_turn", 150, 60)
        # Round 3: end_turn with text AGAIN → falls through (no second recovery)
        round3 = _mock_provider_response("I already told you, 42.", "end_turn", 150, 60)
        provider.call.side_effect = [round1, round2, round3]

        root = Path(tempfile.mkdtemp())
        executor = ToolExecutor(workspace_root=root, task_type=TaskType.COMPUTE)
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.COMPUTER_TOOLS,
            max_rounds=5,
        )

        # Second end_turn falls through to normal return
        assert result.stop_reason == "end_turn"
        assert result.rounds == 3
        assert result.text == "I already told you, 42."


class TestTextEndTurnRecovery:
    """Test text end_turn recovery when model writes findings as text without calling exit tool."""

    @patch("open_dirac.llm._get_provider")
    def test_text_end_turn_recovery_then_exit_tool(self, mock_get_provider):
        """tool_use → text end_turn → recovery → submit_result → executor_stop."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: tool_use with execute_python
        round1 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(42)"}}
            ],
        )
        # Round 2: end_turn with text (model writes analysis instead of calling submit_result)
        round2 = _mock_provider_response(
            "The result is 42, confirmed by numerical computation.",
            "end_turn",
            300,
            100,
        )
        # Round 3: after recovery, model calls submit_result
        round3 = _mock_provider_response(
            "",
            "tool_use",
            150,
            60,
            tool_calls=[
                {
                    "id": "t2",
                    "name": "submit_result",
                    "input": {
                        "target_id": "RQ-001",
                        "description": "result is 42",
                        "method": "numerical",
                        "result": "42",
                        "confidence": "exact",
                        "notes": "done",
                    },
                }
            ],
        )
        provider.call.side_effect = [round1, round2, round3]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=5,
        )

        assert result.stop_reason == "executor_stop"
        assert result.rounds == 3
        assert provider.call.call_count == 3
        # Recovery message was injected
        calls = provider.call.call_args_list
        round3_messages = calls[2].kwargs["messages"]
        recovery_found = any(
            isinstance(msg.get("content"), str)
            and "did not call" in msg["content"]
            and "submit_result" in msg["content"]
            for msg in round3_messages
            if isinstance(msg, dict) and msg.get("role") == "user"
        )
        assert recovery_found, "Recovery message should mention submit_result"

    @patch("open_dirac.llm._get_provider")
    def test_text_end_turn_recovery_no_prior_tools(self, mock_get_provider):
        """text end_turn on round 1 (no prior tools) → immediate end_turn return (no recovery)."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: end_turn with text, no prior tool calls
        round1 = _mock_provider_response("Here is my analysis.", "end_turn", 200, 80)
        provider.call.side_effect = [round1]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=5,
        )

        assert result.stop_reason == "end_turn"
        assert result.rounds == 1
        assert provider.call.call_count == 1  # no recovery

    @patch("open_dirac.llm._get_provider")
    def test_text_end_turn_recovery_second_text_falls_through(self, mock_get_provider):
        """tool_use → text end_turn → recovery → text end_turn again → falls through to end_turn."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: tool_use
        round1 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {"id": "t1", "name": "execute_python", "input": {"code": "print(1)"}}
            ],
        )
        # Round 2: text end_turn → recovery fires
        round2 = _mock_provider_response("The answer is 42.", "end_turn", 150, 60)
        # Round 3: text end_turn AGAIN → falls through (no second recovery)
        round3 = _mock_provider_response("I already said 42.", "end_turn", 150, 60)
        provider.call.side_effect = [round1, round2, round3]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=5,
        )

        assert result.stop_reason == "end_turn"
        assert result.rounds == 3
        assert result.text == "I already said 42."
        assert provider.call.call_count == 3


class TestForcedFinalWithExitTool:
    """Test forced final call keeps exit tool when ready_to_conclude (Part B)."""

    @patch("open_dirac.llm._get_provider")
    def test_forced_final_includes_exit_tool_when_ready(self, mock_get_provider):
        """max_rounds hit + ready_to_conclude → forced call with exit tool → executor_stop."""
        from open_dirac.task import TaskType

        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: report_progress with ready_to_conclude=True
        round1 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {
                    "id": "t1",
                    "name": "report_progress",
                    "input": {
                        "findings_so_far": "Answer found",
                        "remaining_questions": "",
                        "ready_to_conclude": True,
                    },
                }
            ],
        )
        # Round 2: execute_python (model ignores and keeps computing — hits max_rounds)
        round2 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {
                    "id": "t2",
                    "name": "execute_python",
                    "input": {"purpose": "Extra check", "code": "print(1)"},
                }
            ],
        )
        # Forced final call: model calls submit_result via exit tool
        forced_resp = _mock_provider_response(
            "",
            "tool_use",
            150,
            60,
            tool_calls=[
                {
                    "id": "t3",
                    "name": "submit_result",
                    "input": {
                        "target_id": "WH-001",
                        "description": "Entropy",
                        "method": "analytical",
                        "result": "S = k ln(W)",
                        "confidence": "exact",
                        "notes": "Done.",
                    },
                }
            ],
        )
        provider.call.side_effect = [round1, round2, forced_resp]

        root = Path(tempfile.mkdtemp())
        executor = ToolExecutor(workspace_root=root, task_type=TaskType.COMPUTE)
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.COMPUTER_TOOLS,
            max_rounds=2,
        )

        assert result.stop_reason == "executor_stop"
        assert not result.truncated
        # Forced call included the exit tool
        forced_call = provider.call.call_args_list[-1]
        assert forced_call.kwargs.get("tools") is not None
        tool_names = {t["function"]["name"] for t in forced_call.kwargs["tools"]}
        assert tool_names == {"submit_result"}
        # submit_result was actually executed
        assert any(tc.tool_name == "submit_result" for tc in result.tool_calls)

    @patch("open_dirac.llm._get_provider")
    def test_forced_final_always_includes_exit_tool(self, mock_get_provider):
        """max_rounds hit → forced call always includes exit tool, retries if not called."""
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        # Round 1: execute_python (no report_progress, no ready signal)
        round1 = _mock_provider_response(
            "",
            "tool_use",
            200,
            80,
            tool_calls=[
                {
                    "id": "t1",
                    "name": "execute_python",
                    "input": {"purpose": "Check", "code": "print(1)"},
                }
            ],
        )
        # Forced final calls: model returns text (doesn't call exit tool)
        forced1 = _mock_provider_response("Partial.", "end_turn", 150, 60)
        forced2 = _mock_provider_response("More.", "end_turn", 150, 60)
        forced3 = _mock_provider_response("INCONCLUSIVE result.", "end_turn", 150, 60)
        provider.call.side_effect = [round1, forced1, forced2, forced3]

        executor = _make_executor()
        result = run_agent_loop(
            system="sys",
            user_content="question",
            config=_make_config(),
            tool_executor=executor,
            tools=ToolExecutor.TOOL_DEFINITIONS,
            max_rounds=1,
        )

        assert result.stop_reason == "max_rounds_forced"
        assert result.truncated
        # Forced call included exit tool
        forced_call = provider.call.call_args_list[1]
        assert forced_call.kwargs.get("tools") is not None
        tool_names = {t["function"]["name"] for t in forced_call.kwargs["tools"]}
        assert "submit_result" in tool_names
        # 3 forced retries (model never called exit tool)
        assert provider.call.call_count == 4
