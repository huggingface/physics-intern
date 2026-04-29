"""Tests for Task dataclass and TaskType enum."""

from open_dirac.task import Task, TaskType, TASK_TYPE_AGENT_MAP


class TestTaskType:
    def test_values(self):
        assert TaskType.RESEARCH == "research"
        assert TaskType.COMPUTE == "compute"
        assert TaskType.REVIEW == "review"
        assert TaskType.TERMINATE == "terminate"

    def test_from_string(self):
        assert TaskType("compute") == TaskType.COMPUTE
        assert TaskType("research") == TaskType.RESEARCH
        assert TaskType("review") == TaskType.REVIEW


class TestTaskToMarkdown:
    def test_basic(self):
        task = Task(
            task_id="TASK-005",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            priority="high",
            iteration=5,
            body="# Task\n\nVerify result B.",
        )
        md = task.to_markdown()
        assert "task_id: TASK-005" in md
        assert "task_type: review" in md
        assert "assigned_to: reviewer" in md
        assert "Verify result B." in md

    def test_with_blocking_critiques(self):
        task = Task(
            task_id="TASK-010",
            task_type=TaskType.RESEARCH,
            assigned_to="researcher",
            iteration=10,
            blocking_critiques=["CRIT-001", "CRIT-002"],
            body="Resolve critiques.",
        )
        md = task.to_markdown()
        assert "blocking_critiques" in md
        assert "CRIT-001" in md

    def test_structured_dispatch_fields(self):
        task = Task(
            task_id="TASK-015",
            task_type=TaskType.COMPUTE,
            assigned_to="computer",
            iteration=15,
            body="Compute partition function.",
            background="Prior work established ER-001.",
            method_hints=["Use SymPy", "Check limits"],
            assumptions=["T > 0"],
            relevant_results=["ER-001"],
        )
        md = task.to_markdown()
        assert "background:" in md
        assert "method_hints:" in md
        assert "assumptions:" in md
        assert "relevant_results:" in md


class TestTaskFromFrontmatter:
    def test_basic(self):
        text = (
            "---\n"
            "task_id: TASK-003\n"
            "task_type: review\n"
            "assigned_to: reviewer\n"
            "priority: high\n"
            "iteration: 3\n"
            "---\n\n"
            "Verify something."
        )
        task = Task.from_frontmatter(text)
        assert task.task_id == "TASK-003"
        assert task.task_type == TaskType.REVIEW
        assert task.assigned_to == "reviewer"
        assert task.iteration == 3
        assert "Verify something." in task.body

    def test_fallback_iteration(self):
        text = "---\ntask_type: research\nassigned_to: researcher\n---\n\nDo something."
        task = Task.from_frontmatter(text, fallback_iteration=7)
        assert task.task_id == "TASK-007"
        assert task.iteration == 7

    def test_unknown_task_type_defaults_to_research(self):
        text = "---\ntask_type: foobar\nassigned_to: researcher\n---\n\nBody."
        task = Task.from_frontmatter(text)
        assert task.task_type == TaskType.RESEARCH

    def test_round_trip(self):
        original = Task(
            task_id="TASK-042",
            task_type=TaskType.CRITIQUE,
            assigned_to="deep_critic",
            priority="high",
            iteration=42,
            body="Review all claims.",
        )
        md = original.to_markdown()
        restored = Task.from_frontmatter(md)
        assert restored.task_id == original.task_id
        assert restored.task_type == original.task_type
        assert restored.assigned_to == original.assigned_to
        assert restored.iteration == original.iteration
        assert "Review all claims." in restored.body


class TestTaskFromFrontmatterEdgeCases:
    def test_from_frontmatter_empty_assigned_to(self):
        """Empty string assigned_to defaults to 'researcher'."""
        text = "---\ntask_type: review\nassigned_to: ''\niteration: 5\n---\n\nBody."
        task = Task.from_frontmatter(text)
        assert task.assigned_to == "researcher"

    def test_from_frontmatter_null_assigned_to(self):
        """null/None assigned_to defaults to 'researcher'."""
        text = "---\ntask_type: review\nassigned_to:\niteration: 5\n---\n\nBody."
        task = Task.from_frontmatter(text)
        assert task.assigned_to == "researcher"


class TestTaskTypeAgentMap:
    def test_all_task_types_mapped(self):
        for tt in TaskType:
            assert tt in TASK_TYPE_AGENT_MAP, (
                f"TaskType.{tt} not in TASK_TYPE_AGENT_MAP"
            )

    def test_review_maps_to_reviewer(self):
        assert TASK_TYPE_AGENT_MAP[TaskType.REVIEW] == "reviewer"

    def test_critique_maps_to_deep_critic(self):
        assert TASK_TYPE_AGENT_MAP[TaskType.CRITIQUE] == "deep_critic"

    def test_research_maps_to_researcher(self):
        assert TASK_TYPE_AGENT_MAP[TaskType.RESEARCH] == "researcher"

    def test_compute_maps_to_computer(self):
        assert TASK_TYPE_AGENT_MAP[TaskType.COMPUTE] == "computer"


class TestSurveyTaskType:
    def test_survey_task_type_exists(self):
        assert TaskType.SURVEY == "survey"
        assert TaskType("survey") == TaskType.SURVEY

    def test_survey_in_agent_map(self):
        assert TaskType.SURVEY in TASK_TYPE_AGENT_MAP
        assert TASK_TYPE_AGENT_MAP[TaskType.SURVEY] == "surveyor"


class TestTaskTargetClaim:
    def test_target_claim_in_markdown(self):
        task = Task(
            task_id="TASK-005",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="WH-001",
            body="Verify WH-001.",
        )
        md = task.to_markdown()
        assert "target_claim: WH-001" in md

    def test_target_claim_omitted_when_empty(self):
        task = Task(
            task_id="TASK-005",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            body="Verify something.",
        )
        md = task.to_markdown()
        assert "target_claim" not in md

    def test_target_claim_round_trip(self):
        task = Task(
            task_id="TASK-005",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            target_claim="ER-003",
            body="Verify ER-003.",
        )
        restored = Task.from_frontmatter(task.to_markdown())
        assert restored.target_claim == "ER-003"

    def test_target_claim_missing_in_frontmatter(self):
        text = "---\ntask_type: review\nassigned_to: reviewer\n---\n\nBody."
        task = Task.from_frontmatter(text)
        assert task.target_claim == ""
