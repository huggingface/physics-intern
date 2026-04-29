"""Tests verifying agent module structure: imports, tool configurations, and basic process_response."""



class TestNewAgentImports:
    """Verify the new agent modules exist and are importable."""

    def test_researcher_agent_importable(self):
        from open_dirac.agents.researcher import ResearcherAgent

        assert ResearcherAgent.name == "researcher"

    def test_computer_agent_importable(self):
        from open_dirac.agents.computer import ComputerAgent

        assert ComputerAgent.name == "computer"

    def test_verifier_agent_importable(self):
        from open_dirac.agents.reviewer import ReviewerAgent

        assert ReviewerAgent.name == "reviewer"


class TestNewAgentTools:
    """Verify the new agents have correct tool configurations."""

    def test_researcher_is_one_shot(self):
        from open_dirac.agents.researcher import ResearcherAgent

        assert ResearcherAgent.tools == []

    def test_computer_has_tools(self):
        from open_dirac.agents.computer import ComputerAgent

        assert ComputerAgent.tools
        names = {t["function"]["name"] for t in ComputerAgent.tools}
        assert "execute_python" in names
        assert "submit_result" in names

    def test_reviewer_is_one_shot(self):
        from open_dirac.agents.reviewer import ReviewerAgent

        assert ReviewerAgent.tools == []

    def test_critic_is_one_shot(self):
        from open_dirac.agents.critic import CriticAgent

        assert CriticAgent.tools == []

    def test_orchestrator_has_state_mutation_tools(self):
        from open_dirac.agents.orchestrator import OrchestratorAgent

        assert len(OrchestratorAgent.tools) > 0
        tool_names = {t["function"]["name"] for t in OrchestratorAgent.tools}
        assert "dispatch_researcher" in tool_names
        assert "add_hypothesis" in tool_names


class TestComputerProcessResponse:
    """Test ComputerAgent.process_response builds Evidence correctly."""

    def _make_agent(self):
        from open_dirac.agents.computer import ComputerAgent
        from open_dirac.research_state import ResearchState, ResearchQuestion

        agent = ComputerAgent.__new__(ComputerAgent)
        agent.research_state = ResearchState(problem_statement="test")
        rq_id = agent.research_state.next_entity_num()
        agent.research_state.research_questions[rq_id] = ResearchQuestion(
            id=rq_id, question="What is X?"
        )
        agent._last_script_names = []
        return agent

    def _make_result(self, tool_calls):
        from open_dirac.llm import AgentResult

        return AgentResult(text="", tool_calls=tool_calls)

    def _make_tc(self, name, tool_input, output="ok", is_error=False):
        from open_dirac.tool_call import ToolCall

        return ToolCall(
            tool_name=name,
            tool_input=tool_input,
            output=output,
            is_error=is_error,
            duration=0.1,
        )

    def test_approach_includes_assumptions_and_expected_outcome(self):
        from open_dirac.task import Task, TaskType

        agent = self._make_agent()
        rq_id = list(agent.research_state.research_questions.keys())[0]
        task = Task(
            task_id="T1",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            body=f"Compute {rq_id}",
            target_claim=rq_id,
        )
        tool_calls = [
            self._make_tc(
                "document_approach",
                {
                    "approach": "Compute via SymPy",
                    "assumptions": "T > 0. Natural units",
                    "expected_outcome": "Should match Hawking formula",
                },
            ),
            self._make_tc(
                "submit_result",
                {
                    "target_id": rq_id,
                    "method": "symbolic",
                    "result": "T_H = 1/(8*pi*M)",
                    "confidence": "exact",
                },
            ),
        ]
        result = self._make_result(tool_calls)
        agent.process_response(result, task, iteration=1)
        evidence = agent.research_state.research_questions[rq_id].evidence[-1]
        assert evidence is not None
        assert "Compute via SymPy" in evidence.approach
        assert "Assumptions: T > 0. Natural units" in evidence.approach
        assert "Expected outcome: Should match Hawking formula" in evidence.approach

    def test_approach_without_assumptions_or_expected_outcome(self):
        from open_dirac.task import Task, TaskType

        agent = self._make_agent()
        rq_id = list(agent.research_state.research_questions.keys())[0]
        task = Task(
            task_id="T1",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            body=f"Compute {rq_id}",
            target_claim=rq_id,
        )
        tool_calls = [
            self._make_tc("document_approach", {"approach": "Compute via SymPy"}),
            self._make_tc(
                "submit_result",
                {
                    "target_id": rq_id,
                    "method": "symbolic",
                    "result": "42",
                    "confidence": "exact",
                },
            ),
        ]
        result = self._make_result(tool_calls)
        agent.process_response(result, task, iteration=1)
        evidence = agent.research_state.research_questions[rq_id].evidence[-1]
        assert evidence is not None
        assert evidence.approach == "Compute via SymPy"
        assert "Assumptions:" not in evidence.approach
        assert "Expected outcome:" not in evidence.approach


class TestResearcherProcessResponse:
    """Test ResearcherAgent.process_response builds Evidence correctly (one-shot JSON)."""

    def _make_agent(self):
        from open_dirac.agents.researcher import ResearcherAgent
        from open_dirac.research_state import ResearchState, ResearchQuestion

        agent = ResearcherAgent.__new__(ResearcherAgent)
        agent.research_state = ResearchState(problem_statement="test")
        rq_id = f"RQ-{agent.research_state.next_entity_num():03d}"
        agent.research_state.research_questions[rq_id] = ResearchQuestion(
            id=rq_id, question="Derive X?"
        )
        return agent, rq_id

    def _make_response(self, text=""):
        from open_dirac.llm import LLMResponse

        return LLMResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration=0.1,
        )

    def test_evidence_from_json_block(self):
        from open_dirac.task import Task, TaskType

        agent, rq_id = self._make_agent()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body=f"Derive for {rq_id}",
            target_claim=rq_id,
        )
        text = (
            "Full derivation text here...\n\n"
            "```json\n"
            '{"result": "T_H = 1/(8*pi*M)", "method": "Euclidean path integral", '
            '"confidence": "exact", "summary": "Hawking temperature derived"}\n'
            "```"
        )
        response = self._make_response(text=text)
        agent.process_response(response, task, iteration=1)
        evidence = agent.research_state.research_questions[rq_id].evidence[-1]
        assert evidence is not None
        assert evidence.type == "research"
        assert evidence.method == "Euclidean path integral"
        assert evidence.result == "T_H = 1/(8*pi*M)"
        assert evidence.confidence == "exact"
        assert evidence.summary == "Hawking temperature derived"

    def test_reasoning_is_full_response_text(self):
        """Evidence.reasoning is the full response text (derivation + JSON)."""
        from open_dirac.task import Task, TaskType

        agent, rq_id = self._make_agent()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body=f"Derive for {rq_id}",
            target_claim=rq_id,
        )
        derivation_text = (
            "Starting from the metric ds^2 = -(1-2M/r)dt^2 + ... we find...\n\n"
            "```json\n"
            '{"result": "T_H = 1/(8*pi*M)", "method": "analytical", '
            '"confidence": "exact", "summary": "Derived T_H"}\n'
            "```"
        )
        response = self._make_response(text=derivation_text)
        agent.process_response(response, task, iteration=1)
        evidence = agent.research_state.research_questions[rq_id].evidence[-1]
        assert "Starting from the metric" in evidence.reasoning

    def test_target_from_task_target_claim(self):
        """Target ID comes from task.target_claim."""
        from open_dirac.task import Task, TaskType

        agent, rq_id = self._make_agent()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body="Some task",
            target_claim=rq_id,
        )
        text = '```json\n{"result": "42", "method": "m", "confidence": "exact", "summary": "s"}\n```'
        response = self._make_response(text=text)
        agent.process_response(response, task, iteration=1)
        assert len(agent.research_state.research_questions[rq_id].evidence) > 0

    def test_fallback_no_json(self):
        """When no JSON block, raise ParseFailureError (no degraded evidence)."""
        import pytest
        from open_dirac.llm import ParseFailureError
        from open_dirac.task import Task, TaskType

        agent, rq_id = self._make_agent()
        task = Task(
            task_id="T1",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            body=f"Derive for {rq_id}",
            target_claim=rq_id,
        )
        response = self._make_response(text="Partial derivation that got cut off...")
        with pytest.raises(ParseFailureError):
            agent.process_response(response, task, iteration=1)
        assert len(agent.research_state.research_questions[rq_id].evidence) == 0


class TestToolsForTaskType:
    """Test tools_for_task_type returns correct tool sets."""

    def test_research_tools(self):
        from open_dirac.task import TaskType
        from open_dirac.agents.computer.tools import ToolExecutor

        names = {
            t["function"]["name"]
            for t in ToolExecutor.tools_for_task_type(TaskType.RESEARCH)
        }
        assert "submit_result" in names

    def test_compute_tools(self):
        from open_dirac.task import TaskType
        from open_dirac.agents.computer.tools import ToolExecutor

        names = {
            t["function"]["name"]
            for t in ToolExecutor.tools_for_task_type(TaskType.COMPUTE)
        }
        assert "execute_python" in names
        assert "submit_result" in names
