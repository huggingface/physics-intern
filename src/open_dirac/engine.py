"""OpenDirac main loop engine."""

from pathlib import Path

from rich.panel import Panel

from .config import Config
from .console import console, replay_log
from .console_reports import (
    on_round_progress,
    print_call_summary,
    print_final_report,
    print_task,
)
from .critique_routing import (
    adjudicate_er_critique,
    auto_promote,
    invoke_planner_revision,
    route_critiques,
)
from .dispatcher import (
    dispatch as _dispatcher_dispatch,
    handle_context_too_long as _dispatcher_handle_context_too_long,
    handle_dispatch_error as _dispatcher_handle_dispatch_error,
    handle_parse_failure as _dispatcher_handle_parse_failure,
    is_recoverable,
    record_agent_failures as _dispatcher_record_agent_failures,
)
from .llm import AgentResult, LLMResponse, ParseFailureError
from .loop_state import (
    LoopState,
    append_dispatch_record,
    build_context_suffix,
    render_pending_work,
)
from .providers import ContextTooLongError
from .metrics import MetricsTracker
from .resume import (
    find_last_critic_iteration as _find_last_critic_iteration,
    reconstruct_loop_state as _reconstruct_loop_state,
)
from .task import Task, TaskType
from .utils.categories import CompensationCategory as CC
from .research_state import ResearchState, Verdict
from .state_transitions import normalize_references
from .rendering import (
    render_research_state_md,
    render_evidence_log_md,
    render_critique_log_md,
)
from .validation import (
    validate_post_integration,
    can_terminate,
    Violation,
    ViolationSeverity,
)
from .workspace import WorkspaceManager, log_scaffold_event
from .agents.orchestrator import OrchestratorAgent
from .agents.researcher import ResearcherAgent
from .agents.computer import ComputerAgent
from .agents.reviewer import ReviewerAgent
from .agents.critic import CriticAgent
from .agents.formatter import FormatterAgent
from .agents.surveyor import SurveyorAgent
from .agents.planner import PlannerAgent
from .agents.adjudicator import AdjudicatorAgent
from .verification import (
    run_formal_evaluation,
    render_formal_evaluation,
    write_formal_eval_report,
)


_is_recoverable = is_recoverable  # backward-compat alias for local use


class OpenDirac:
    """Main loop for the OpenDirac research system."""

    def __init__(
        self,
        problem: str,
        config: Config | None = None,
        problem_meta: dict | None = None,
        answer_template: str = "",
    ):
        self.config = config or Config()
        self.metrics = MetricsTracker()
        self.workspace = WorkspaceManager(self.config)
        self.workspace.init(problem)
        self.config.logs_dir = str(self.workspace.logs_dir)
        console.setup_log(Path(self.workspace.logs_dir) / "console.log")
        self.iteration = 0
        self._state = LoopState()
        self.research_state = ResearchState()
        self.research_state.problem_statement = problem.strip()
        self.research_state.answer_template = (
            answer_template.strip() if answer_template else ""
        )
        self.research_state.title = self.workspace.root.name
        self.problem_meta = problem_meta or {}

        # Initialize agents
        self.orchestrator = OrchestratorAgent(self.config, self.workspace, self.metrics)
        self.researcher = ResearcherAgent(self.config, self.workspace, self.metrics)
        self.computer = ComputerAgent(self.config, self.workspace, self.metrics)
        self.reviewer = ReviewerAgent(self.config, self.workspace, self.metrics)
        self.critic = CriticAgent(self.config, self.workspace, self.metrics)
        self.formatter = FormatterAgent(
            self.config, self.workspace, self.metrics, answer_template
        )
        self.surveyor = SurveyorAgent(self.config, self.workspace, self.metrics)
        self.planner = PlannerAgent(self.config, self.workspace, self.metrics)
        self.adjudicator = AdjudicatorAgent(self.config, self.workspace, self.metrics)

    @classmethod
    def resume(
        cls,
        workspace_path: Path | str,
        config_overrides: dict | None = None,
        answer_template: str = "",
    ) -> "OpenDirac":
        """Resume a previously interrupted run from its last committed state."""
        import yaml as _yaml

        workspace_path = Path(workspace_path)

        # 1. Load config
        config = Config.load(workspace_path, overrides=config_overrides)
        config.workspace_dir = str(workspace_path)

        # 2. Load problem.yaml from workspace
        problem_yaml_path = workspace_path / "problem.yaml"
        if not problem_yaml_path.exists():
            raise FileNotFoundError(
                f"No problem.yaml in {workspace_path} — cannot resume "
                f"(workspace predates the resume feature?)"
            )
        with open(problem_yaml_path) as f:
            problem_def = _yaml.safe_load(f)
        problem_meta = {
            "steps": problem_def.get("steps", []),
        }
        if not answer_template:
            answer_template = problem_def.get("answer_template", "")

        # 3. Build engine without calling __init__
        engine = cls.__new__(cls)
        engine.config = config
        engine.metrics = MetricsTracker.load(workspace_path)
        engine.workspace = WorkspaceManager(config)
        engine.workspace.attach()
        engine.config.logs_dir = str(engine.workspace.logs_dir)
        log_path = Path(engine.workspace.logs_dir) / "console.log"
        replay_log(log_path)
        console.setup_log(log_path)
        engine.problem_meta = problem_meta

        # 4. Load research state
        engine.research_state = ResearchState.load(workspace_path)
        engine.iteration = engine.research_state.iteration
        # Ensure answer_template is populated (backward compat: old states lack this field)
        if not engine.research_state.answer_template and answer_template:
            engine.research_state.answer_template = answer_template.strip()

        # 5. Reconstruct loop state
        engine._state = _reconstruct_loop_state(engine.research_state)

        # 6. Reconstruct last critic iteration
        engine.metrics.last_critic_iteration = _find_last_critic_iteration(
            workspace_path
        )

        # 7. Initialize agents (same as __init__)
        engine.orchestrator = OrchestratorAgent(
            config, engine.workspace, engine.metrics
        )
        engine.researcher = ResearcherAgent(config, engine.workspace, engine.metrics)
        engine.computer = ComputerAgent(config, engine.workspace, engine.metrics)
        engine.reviewer = ReviewerAgent(config, engine.workspace, engine.metrics)
        engine.critic = CriticAgent(config, engine.workspace, engine.metrics)
        engine.formatter = FormatterAgent(
            config, engine.workspace, engine.metrics, answer_template
        )
        engine.surveyor = SurveyorAgent(config, engine.workspace, engine.metrics)
        engine.planner = PlannerAgent(config, engine.workspace, engine.metrics)
        engine.adjudicator = AdjudicatorAgent(config, engine.workspace, engine.metrics)

        console.print(
            Panel(
                f"Resuming from iteration {engine.iteration}",
                style="bold blue",
            )
        )
        return engine

    def run(self):
        """Main loop: survey → orchestrate → validate → override → dispatch → git."""
        console.print(Panel("OpenDirac Research System", style="bold blue"))

        # Skip surveyor if survey fields already populated (e.g. on resume)
        if not self.research_state.survey_background:
            self._run_with_pipeline_retry("Surveyor", self._run_surveyor)
        else:
            console.print(
                "[dim]Surveyor skipped (background survey already exists)[/dim]"
            )

        # Run planner to produce initial strategy (skip if already set, e.g. on resume)
        if not self.research_state.strategy:
            self._run_with_pipeline_retry("Planner", self._run_planner)
        else:
            console.print("[dim]Planner skipped (strategy already exists)[/dim]")

        if self.research_state.status == "completed":
            console.print("[yellow]This workspace is already completed.[/yellow]")
            return

        while self.iteration < self.config.max_iterations:
            self.iteration += 1
            console.rule(f"[bold]ITERATION {self.iteration}[/bold]")
            self._update_research_iteration()
            self._auto_expire_critiques()

            # 1. Orchestrator pass
            try:
                task = self._run_orchestrator()
            except ContextTooLongError as exc:
                console.print(
                    f"[yellow]Orchestrator context too long — skipping to "
                    f"next iteration: {exc}[/yellow]"
                )
                self.metrics.alert(
                    self.iteration,
                    f"Orchestrator context too long: {exc.input_tokens} input tokens",
                )
                log_scaffold_event(
                    self.workspace.root,
                    self.iteration,
                    CC.LOOP_CONTROL,
                    "context_too_long",
                    f"agent=orchestrator, input_tokens={exc.input_tokens}",
                )
                continue
            except Exception as exc:
                if not _is_recoverable(exc):
                    raise
                self.metrics.alert(
                    self.iteration,
                    f"Orchestrator failed (recoverable): {type(exc).__name__}: {exc}",
                )
                self._state.pending_violations.append(
                    Violation(
                        check="orchestrator_failure",
                        severity=ViolationSeverity.WARNING,
                        message=(
                            f"Orchestrator failed (recoverable): "
                            f"{type(exc).__name__}: {str(exc)[:200]}"
                        ),
                    )
                )
                console.print(
                    f"[yellow]Orchestrator failed (recoverable), skipping to "
                    f"next iteration: {exc}[/yellow]"
                )
                log_scaffold_event(
                    self.workspace.root,
                    self.iteration,
                    CC.LOOP_CONTROL,
                    "orchestrator_failure",
                    f"{type(exc).__name__}: {str(exc)[:200]}",
                )
                continue

            # 2. Post-integration validation
            violations = validate_post_integration(
                self.research_state,
                iteration=self.iteration,
                workspace=self.workspace,
            )
            if violations:
                self._state.pending_violations.extend(violations)

            # 3. Enrichment for compute tasks (best-effort — never fatal)
            if task.task_type == TaskType.COMPUTE:
                try:
                    self._enrich_compute_task_with_prior_failures(task)
                except Exception as exc:
                    console.print(
                        f"[yellow]Compute enrichment failed (non-fatal): "
                        f"{type(exc).__name__}: {exc}[/yellow]"
                    )
                    log_scaffold_event(
                        self.workspace.root,
                        self.iteration,
                        CC.LOOP_CONTROL,
                        "enrichment_error",
                        f"{type(exc).__name__}: {str(exc)[:200]}",
                    )

            # 4. Termination gate
            if task.task_type == TaskType.TERMINATE:
                allowed, blockers = can_terminate(
                    self.workspace,
                    self.config,
                    self.metrics,
                    self.problem_meta,
                    research_state=self.research_state,
                )
                if allowed:
                    console.print("[green]Orchestrator signaled completion.[/green]")
                    rejection = self._run_formatter(answer_ers=task.answer_ers)
                    if rejection is None:
                        self._set_research_status("completed")
                        self._render_files_for_git()
                        self._sync_research_state()
                        break
                    # Formatter rejected — loop back to orchestrator
                    self._state.consecutive_termination_blocks += 1
                    blockers = [
                        f"FORMATTER REJECTED the answer: {rejection}. "
                        "The established results do not contain concrete "
                        "values matching the answer template. Dispatch "
                        "research or compute tasks to derive the missing "
                        "concrete results before terminating again."
                    ]
                    log_scaffold_event(
                        self.workspace.root,
                        self.iteration,
                        CC.LOOP_CONTROL,
                        "formatter_rejection_loopback",
                        f"consecutive={self._state.consecutive_termination_blocks}, "
                        f"reason={rejection[:200]}",
                    )
                    if (
                        self._state.consecutive_termination_blocks
                        >= self.config.max_termination_retries
                    ):
                        console.print(
                            "[yellow]Circuit breaker: accepting best-effort "
                            "formatter output despite rejection[/yellow]"
                        )
                        self._render_files_for_git()
                        self.workspace.git_commit(
                            f"Iteration {self.iteration}: formatter - "
                            "ANSWER.md (best-effort, rejected)"
                        )
                        self._set_research_status("partially_complete")
                        self._render_files_for_git()
                        self._sync_research_state()
                        break
                else:
                    self._state.consecutive_termination_blocks += 1
                    blockers = list(blockers)
                self._state.pending_termination_blockers = blockers
                log_scaffold_event(
                    self.workspace.root,
                    self.iteration,
                    CC.LOOP_CONTROL,
                    "termination_blocked",
                    f"blockers: {'; '.join(b[:60] for b in blockers)}, "
                    f"consecutive={self._state.consecutive_termination_blocks}",
                )
                # Circuit breaker: after repeated blocks, auto-abandon remaining WHs
                if (
                    self._state.consecutive_termination_blocks
                    >= self.config.max_termination_retries
                ):
                    self._force_abandon_working_hypotheses()
                self._append_dispatch_record(task)
                continue  # re-enter loop
            else:
                self._state.consecutive_termination_blocks = 0

            # 5. Dispatch to agent
            try:
                agent_name, agent_result = self._dispatch(task)
            except ContextTooLongError as exc:
                self._handle_context_too_long(task, exc)
                continue
            except ParseFailureError as exc:
                self._handle_parse_failure(task, exc)
                continue
            except Exception as exc:
                if not _is_recoverable(exc):
                    raise
                self._handle_dispatch_error(task, exc)
                continue

            # 5b. Record agent failure signals for orchestrator context
            self._record_agent_failures(task, agent_name, agent_result)

            # 6. Post-dispatch checks
            if task.task_type in (TaskType.RESEARCH, TaskType.COMPUTE, TaskType.REVIEW):
                self._track_agent_result(task)
            self._append_dispatch_record(task)

            # 6b. Auto-review for WHs with new evidence after stale review
            auto_review_target = self._should_auto_review(task)
            if auto_review_target:
                console.print(
                    f"[yellow]Auto-dispatching reviewer for {auto_review_target} "
                    f"(new evidence after stale review)[/yellow]"
                )
                log_scaffold_event(
                    self.workspace.root,
                    self.iteration,
                    CC.LOOP_CONTROL,
                    "auto_review",
                    f"target={auto_review_target}, trigger=new_evidence",
                )
                review_task = self._make_auto_review_task(auto_review_target)
                try:
                    agent_name_r, result_r = self._dispatch(review_task)
                    self._record_agent_failures(review_task, agent_name_r, result_r)
                    self._track_agent_result(review_task)
                    self._append_dispatch_record(review_task)
                except ContextTooLongError as exc:
                    self._handle_context_too_long(review_task, exc)
                except ParseFailureError as exc:
                    self._handle_parse_failure(review_task, exc)
                except Exception as exc:
                    if not _is_recoverable(exc):
                        raise
                    self._handle_dispatch_error(review_task, exc)

            # 6c. Auto-trigger critic after VERIFIED review
            if self._should_trigger_critic():
                console.print(
                    f"[yellow]Auto-triggering critic after VERIFIED review "
                    f"(last critic at iter {self.metrics.last_critic_iteration})[/yellow]"
                )
                log_scaffold_event(
                    self.workspace.root,
                    self.iteration,
                    CC.LOOP_CONTROL,
                    "forced_critic",
                    f"trigger=verified_review, "
                    f"last_critic={self.metrics.last_critic_iteration}",
                )
                critic_task = self._make_forced_critic_task()
                try:
                    agent_name_c, result_c = self._dispatch(critic_task)
                    self._record_agent_failures(critic_task, agent_name_c, result_c)
                    self._append_dispatch_record(critic_task)
                except ContextTooLongError as exc:
                    self._handle_context_too_long(critic_task, exc)
                except ParseFailureError as exc:
                    self._handle_parse_failure(critic_task, exc)
                except Exception as exc:
                    if not _is_recoverable(exc):
                        raise
                    self._handle_dispatch_error(critic_task, exc)

            # 6d. Periodic critic safeguard (no verified review required)
            elif self._should_force_periodic_critic():
                console.print(
                    f"[yellow]Forcing periodic critic — no critic in "
                    f"{self.iteration - self.metrics.last_critic_iteration} iterations "
                    f"(safeguard: critic_every_n={self.config.critic_every_n})[/yellow]"
                )
                log_scaffold_event(
                    self.workspace.root,
                    self.iteration,
                    CC.LOOP_CONTROL,
                    "forced_critic",
                    f"trigger=periodic_safeguard, "
                    f"last_critic={self.metrics.last_critic_iteration}",
                )
                critic_task = self._make_forced_critic_task()
                try:
                    agent_name_c, result_c = self._dispatch(critic_task)
                    self._record_agent_failures(critic_task, agent_name_c, result_c)
                    self._append_dispatch_record(critic_task)
                except ContextTooLongError as exc:
                    self._handle_context_too_long(critic_task, exc)
                except ParseFailureError as exc:
                    self._handle_parse_failure(critic_task, exc)
                except Exception as exc:
                    if not _is_recoverable(exc):
                        raise
                    self._handle_dispatch_error(critic_task, exc)

            # 6e. Route critic findings to specialist agents
            self._route_critiques()

            # 7. Metrics, structured state snapshot, render files, git
            self._update_metrics()
            self._sync_research_state()
            self._render_files_for_git()
            self.workspace.git_commit(
                f"Iteration {self.iteration}: {agent_name} - {task.task_id}"
            )

            # 8. Post-dispatch status check (safety net)
            if self._check_status_field():
                console.print("[green]Research completed or abandoned.[/green]")
                break

        # Best-effort formatter when loop exhausted without completion
        if self.iteration > 0 and self.research_state.status not in (
            "completed",
            "partially_complete",
            "abandoned",
        ):
            console.print(
                "[yellow]Max iterations reached without completion — "
                "attempting best-effort formatter.[/yellow]"
            )
            self._set_research_status("partially_complete")
            self.formatter.best_effort = True
            try:
                rejection = self._run_formatter(answer_ers=None)
                if rejection:
                    console.print(
                        f"[yellow]Best-effort formatter rejected: {rejection}[/yellow]"
                    )
            except Exception as exc:
                console.print(
                    f"[yellow]Best-effort formatter failed: "
                    f"{type(exc).__name__}: {exc}[/yellow]"
                )
            finally:
                self.formatter.best_effort = False
            self._render_files_for_git()
            self._sync_research_state()
            self.workspace.git_commit(
                f"Iteration {self.iteration}: best-effort formatter "
                "(max iterations reached)"
            )

        self._final_report()
        self._run_formal_verification()

    def _run_formal_verification(self):
        """Run formal (symbolic/numerical) answer evaluation at end of run."""
        import yaml

        problem_path = self.workspace.root / "problem.yaml"
        if not problem_path.exists():
            console.print(
                "[dim]Formal verification skipped: no problem.yaml in workspace[/]"
            )
            return

        try:
            with open(problem_path) as f:
                problem_def = yaml.safe_load(f)
        except Exception as exc:
            console.print(
                f"[yellow]Formal verification skipped: could not read problem.yaml: {exc}[/]"
            )
            return

        # Build reference lookup path from problem name
        problem_name = problem_def.get("name") if problem_def else None
        ref_lookup_path = Path(problem_name + ".yaml") if problem_name else None

        console.print("\n[bold]Formal answer evaluation...[/]")

        try:
            result = run_formal_evaluation(
                str(self.workspace.root),
                problem_def,
                problem_path=ref_lookup_path,
            )
            render_formal_evaluation(result)
            write_formal_eval_report(result, str(self.workspace.root))
            self.workspace.git_commit("Formal answer evaluation")
        except Exception as exc:
            console.print(
                f"[yellow]Formal verification failed: {type(exc).__name__}: {exc}[/]"
            )

    def _run_orchestrator(self) -> Task:
        """Run orchestrator pass: set context prefix, get task."""
        console.print("[cyan]Orchestrator[/cyan] planning...")

        # Set context prefix for violations/blockers
        self.orchestrator.context_suffix = self._build_context_suffix()
        # Pass research state reference for tool executor and context rendering
        self.orchestrator.research_state = self.research_state

        orch_task = Task(
            task_id="",
            task_type=TaskType.RESEARCH,
            assigned_to="orchestrator",
            iteration=self.iteration,
        )
        orch_response = self.orchestrator.run(
            orch_task, self.iteration, on_round=self._on_agent_round
        )
        task = self.orchestrator.parse_task(
            orch_response.text, iteration=self.iteration
        )
        self._print_task(task)
        return task

    def _run_surveyor(self):
        """Run surveyor agent before the main loop to produce a background survey."""
        console.print("[cyan]Surveyor[/cyan] analyzing problem...")
        self.surveyor.research_state = self.research_state
        task = Task(
            task_id="SURVEY-000",
            task_type=TaskType.SURVEY,
            assigned_to="surveyor",
            iteration=0,
        )
        result = self.surveyor.run(task, 0)
        self._print_call_summary(result)
        self._apply_survey()
        self._sync_research_state()
        self._render_files_for_git()
        self.workspace.git_commit("Iteration 0: surveyor — background survey")

    def _apply_survey(self):
        """Distribute the surveyor's parsed survey into research state fields."""
        survey = self.surveyor.parsed_survey
        if not survey:
            return

        # §1 (background) and §2 (key insights) stored as separate fields
        if survey.get("background"):
            self.research_state.survey_background = survey["background"]
        elif survey.get("raw_notes"):
            # Fallback: no structured sections, use raw text
            self.research_state.survey_background = survey["raw_notes"]
        if survey.get("key_insights"):
            self.research_state.key_insights = survey["key_insights"]

        if survey.get("known_methods"):
            self.research_state.survey_methods = survey["known_methods"]
        if survey.get("known_pitfalls"):
            self.research_state.known_pitfalls = survey["known_pitfalls"]
        if (
            survey.get("conventions_and_definitions")
            and not self.research_state.conventions
        ):
            self.research_state.conventions = survey[
                "conventions_and_definitions"
            ].strip()
        if survey.get("sanity_checks") and not self.research_state.sanity_checks:
            from .research_state import SanityCheck

            for item in survey["sanity_checks"]:
                sc_num = self.research_state.next_sc_num()
                if isinstance(item, dict):
                    self.research_state.sanity_checks.append(
                        SanityCheck(
                            id=f"SC-{sc_num:03d}",
                            predicate=item.get("predicate", str(item)),
                            rationale=item.get("rationale", ""),
                        )
                    )
                else:
                    self.research_state.sanity_checks.append(
                        SanityCheck(
                            id=f"SC-{sc_num:03d}",
                            predicate=str(item),
                        )
                    )
        if survey.get("expected_answer_structure"):
            self.research_state.expected_answer_structure = survey[
                "expected_answer_structure"
            ].strip()
        if survey.get("problem_summary"):
            self.research_state.problem_summary = survey["problem_summary"].strip()

    def _run_planner(self):
        """Run planner agent to produce an initial research strategy."""
        console.print("[cyan]Planner[/cyan] formulating strategy...")
        self.planner.research_state = self.research_state
        task = Task(
            task_id="PLAN-000",
            task_type=TaskType.PLAN,
            assigned_to="planner",
            iteration=0,
        )
        result = self.planner.run(task, 0)
        self._print_call_summary(result)
        self._apply_strategy()
        self._sync_research_state()
        self._render_files_for_git()
        self.workspace.git_commit("Iteration 0: planner — research strategy")

    def _apply_strategy(self):
        """Store the planner's parsed strategy in research state."""
        strategy = self.planner.parsed_strategy
        if strategy is None:
            return

        self.research_state.strategy = strategy

    def _handle_context_too_long(self, task: Task, exc: ContextTooLongError) -> None:
        """Log context-too-long error and record for orchestrator — do not crash."""
        _dispatcher_handle_context_too_long(
            task,
            exc,
            self.iteration,
            self._state,
            self.workspace,
            self.metrics,
        )

    def _handle_dispatch_error(self, task: Task, exc: Exception) -> None:
        """Handle transient dispatch errors — log and skip to next iteration."""
        _dispatcher_handle_dispatch_error(
            task,
            exc,
            self.iteration,
            self._state,
            self.workspace,
            self.metrics,
        )

    def _handle_parse_failure(self, task: Task, exc: ParseFailureError) -> None:
        """Handle evidence agent parse failure — log and report to orchestrator."""
        _dispatcher_handle_parse_failure(
            task,
            exc,
            self.iteration,
            self._state,
            self.workspace,
            self.metrics,
        )

    def _run_with_pipeline_retry(self, label: str, fn) -> None:
        """Run a mandatory pipeline stage with retry on transient errors.

        Surveyor and planner are required before the main loop.  If retries
        are exhausted the entire run is aborted with a clear error message.
        """
        import time

        max_retries = self.config.pipeline_retry_max
        delay = self.config.api_retry_initial_delay
        for attempt in range(max_retries + 1):
            try:
                fn()
                return
            except ContextTooLongError:
                raise  # Deterministic — retrying won't help
            except Exception as exc:
                if not _is_recoverable(exc) or attempt == max_retries:
                    console.print(
                        f"[red]{label} failed after {attempt + 1} attempt(s): "
                        f"{type(exc).__name__}: {exc}[/red]"
                    )
                    raise RuntimeError(
                        f"{label} failed — provider may be unavailable. "
                        f"Last error: {type(exc).__name__}: {exc}"
                    ) from exc
                console.print(
                    f"[yellow]{label} failed (attempt {attempt + 1}/"
                    f"{max_retries + 1}), retrying: {exc}[/yellow]"
                )
                log_scaffold_event(
                    self.workspace.root,
                    0,
                    CC.CALL_RELIABILITY,
                    f"{label.lower()}_retry",
                    f"attempt={attempt + 1}, {type(exc).__name__}: {str(exc)[:200]}",
                )
                time.sleep(min(delay, self.config.api_retry_max_delay))
                delay *= 2

    def _record_agent_failures(self, task: Task, agent_name: str, result):
        """Inspect agent result for failure signals and record for orchestrator context."""
        _dispatcher_record_agent_failures(
            task,
            agent_name,
            result,
            self.iteration,
            self._state,
            self.workspace,
        )

    def _append_dispatch_record(self, task: Task):
        """Derive outcome from authoritative state and append a DispatchRecord."""
        append_dispatch_record(self._state, self.research_state, task, self.iteration)

    def _build_context_suffix(self) -> str:
        """Build suffix for orchestrator context and stash dispatch history on orchestrator."""
        suffix, dispatch_history_text = build_context_suffix(
            self._state,
            self.research_state,
            self.config,
            self.iteration,
        )
        if dispatch_history_text:
            self.orchestrator.dispatch_history_text = dispatch_history_text
        return suffix

    def _render_pending_work(self) -> str:
        """Render a summary of open RQs, working WHs, and dangling WHs."""
        return render_pending_work(self.research_state)

    def _dispatch(self, task: Task) -> tuple[str, LLMResponse | AgentResult]:
        """Route task to the correct agent. Returns (agent_name, result)."""
        return _dispatcher_dispatch(
            task,
            self.iteration,
            self.research_state,
            self._state,
            researcher=getattr(self, "researcher", None),
            computer=getattr(self, "computer", None),
            reviewer=getattr(self, "reviewer", None),
            critic=getattr(self, "critic", None),
            formatter=getattr(self, "formatter", None),
            on_compute_round=self._on_compute_round,
            on_agent_round=self._on_agent_round,
            print_call_summary=self._print_call_summary,
        )

    def _should_trigger_critic(self) -> bool:
        """Check if the critic should auto-trigger after a VERIFIED review."""
        # Trigger on every established result, no delay constraint
        return self._state.last_verified_review_iteration == self.iteration

    def _should_force_periodic_critic(self) -> bool:
        """Force critic if it hasn't run in critic_every_n iterations.

        Safeguard against runs where no VERIFIED review happens for a long
        time — the critic still gets to review strategy and research direction.
        """
        gap = self.iteration - self.metrics.last_critic_iteration
        return gap >= self.config.critic_every_n

    # ------------------------------------------------------------------
    # Critique routing (synchronous, between iterations)
    # ------------------------------------------------------------------

    def _route_critiques(self) -> None:
        """Route critic findings to specialist agents (synchronous).

        Phase 1: Adjudicate ER-targeted critiques via the adjudicator.
        Phase 2: Route strategy/coordination critiques (and any ER demotions
                 from phase 1) to the planner for strategy revision.
        """
        route_critiques(
            self.research_state,
            self._state,
            self.iteration,
            self.workspace,
            self.planner,
            self.adjudicator,
            self._on_agent_round,
        )

    def _adjudicate_er_critique(self, crit) -> dict | None:
        """Invoke the adjudicator to evaluate an ER-targeted critique."""
        return adjudicate_er_critique(
            crit,
            self.research_state,
            self._state,
            self.iteration,
            self.workspace,
            self.adjudicator,
            self._on_agent_round,
        )

    def _invoke_planner_revision(self, strategy_critiques, er_demotions):
        """Invoke the planner in revise mode to assess strategy after critiques/demotions."""
        invoke_planner_revision(
            strategy_critiques,
            er_demotions,
            self.research_state,
            self._state,
            self.iteration,
            self.workspace,
            self.planner,
            self._on_agent_round,
        )

    def _auto_promote(self, wh_id: str):
        """Auto-promote a VERIFIED WH to ER if dependencies are satisfied."""
        auto_promote(self.research_state, wh_id, self.iteration, self.workspace)

    def _make_forced_critic_task(self) -> Task:
        """Create a forced critic task."""
        task = Task(
            task_id=f"TASK-{self.iteration:03d}",
            task_type=TaskType.CRITIQUE,
            assigned_to="deep_critic",
            priority="high",
            iteration=self.iteration,
            body=(
                "# Task Description\n\n"
                "Mandatory periodic review. Perform a thorough critique of all Working\n"
                "Hypotheses and recent Established Results in RESEARCH_STATE.md.\n"
            ),
        )
        self.workspace.write_file("CURRENT_TASK.md", task.to_markdown())
        return task

    def _should_auto_review(self, task: Task) -> str | None:
        """Check if a WH needs auto-review after new evidence from this task.

        Returns the WH ID to review, or None.
        Triggers when a researcher/computer deposits evidence on a WH that
        already has a review older than the newest evidence (re-review cycle).
        """
        if task.task_type not in (TaskType.RESEARCH, TaskType.COMPUTE):
            return None
        target_id = task.target_claim
        if not target_id or not target_id.startswith("WH-"):
            return None
        h = self.research_state.hypotheses.get(target_id)
        if not h or not h.evidence or not h.review:
            return None
        from .research_state import HypothesisStatus

        if h.status == HypothesisStatus.ABANDONED:
            return None
        # Check if any evidence is newer than the review
        review_iter = h.review.iteration or 0
        newest_evidence_iter = max((ev.iteration or 0) for ev in h.evidence)
        if newest_evidence_iter > review_iter:
            return target_id
        return None

    def _make_auto_review_task(self, wh_id: str) -> Task:
        """Create an auto-review task for a WH."""
        task = Task(
            task_id=f"TASK-{self.iteration:03d}",
            task_type=TaskType.REVIEW,
            assigned_to="reviewer",
            priority="high",
            iteration=self.iteration,
            target_claim=wh_id,
            body=f"Auto-triggered review of {wh_id}.",
        )
        self.workspace.write_file("CURRENT_TASK.md", task.to_markdown())
        return task

    def _enrich_compute_task_with_prior_failures(self, task: Task):
        """Append prior failure context to CURRENT_TASK.md for compute retries."""
        import re as _re

        target_ids = set(_re.findall(r"(?:ER|WH)-\d+", task.body or ""))
        if task.target_claim:
            target_ids.update(_re.findall(r"(?:ER|WH)-\d+", task.target_claim))
        if not target_ids:
            return

        # Check claim failure count for these targets
        failure_count = sum(
            self._state.claim_failure_count.get(tid, 0) for tid in target_ids
        )
        if not failure_count:
            return

        log_scaffold_event(
            self.workspace.root,
            self.iteration,
            CC.LOOP_CONTROL,
            "compute_enrichment",
            f"target={','.join(sorted(target_ids))}, failures={failure_count}",
        )
        task_text = self.workspace.read_file("CURRENT_TASK.md")
        addendum = (
            "\n\n---\n\n## Prior Failure Context\n\n"
            f"**{failure_count} prior failure(s) on this claim.** "
            "Diagnose the ROOT CAUSE before writing new code.\n"
        )
        self.workspace.write_file("CURRENT_TASK.md", task_text + addendum)

    def _track_agent_result(self, task: Task):
        """After researcher/computer/reviewer runs, track results for orchestrator banners."""
        tt = task.task_type
        target_id = task.target_claim

        if tt in (TaskType.RESEARCH, TaskType.COMPUTE):
            # Find evidence on the target entity
            ev = None
            evs: list = []
            if target_id in self.research_state.research_questions:
                evs = self.research_state.research_questions[target_id].evidence
                ev = evs[-1] if evs else None
            elif target_id in self.research_state.hypotheses:
                evs = self.research_state.hypotheses[target_id].evidence
                ev = evs[-1] if evs else None
            elif target_id in self.research_state.critiques:
                evs = self.research_state.critiques[target_id].evidence
                ev = evs[-1] if evs else None

            if ev and ev.result:
                # Compute evidence metadata for banner nudges
                active_evs = [e for e in evs if not e.refuted]
                type_counts: dict[str, int] = {}
                for e in active_evs:
                    type_counts[e.type] = type_counts.get(e.type, 0) + 1

                description = ev.summary or (
                    ev.method[:500] if ev.method else "unknown"
                )
                self._state.pending_explore_results.append(
                    {
                        "target_id": target_id,
                        "description": description,
                        "result": ev.result,
                        "confidence": ev.confidence or "partial",
                        "task_id": task.task_id,
                        "task_type": task.task_type.value,
                        "evidence_id": ev.id,
                        "evidence_count": len(active_evs),
                        "evidence_types": type_counts,
                        "target_is_rq": target_id.startswith("RQ-"),
                    }
                )
                snippet = (ev.summary or ev.result[:120]).replace("\n", " ")
                conf = f", {ev.confidence}" if ev.confidence else ""
                ev_tag = f" [{ev.id}]" if ev.id else ""
                console.print(
                    f"  [blue]Evidence for {target_id}[/blue]{ev_tag}{conf}: {snippet}"
                )
            else:
                log_scaffold_event(
                    self.workspace.root,
                    self.iteration,
                    CC.LOOP_CONTROL,
                    "evidence_suppressed",
                    f"target={target_id}",
                )
                console.print(f"  [dim]No evidence produced for {target_id}[/dim]")

        elif tt == TaskType.REVIEW:
            if not target_id or target_id not in self.research_state.hypotheses:
                return
            h = self.research_state.hypotheses[target_id]
            if not h.review:
                return
            verdict = h.review.verdict
            if verdict == Verdict.VERIFIED:
                self._state.pending_verified_results.append(
                    {
                        "claim": target_id,
                        "verdict": verdict,
                        "task_id": task.task_id,
                        "reasoning": h.review.summary or "",
                    }
                )
                self._state.claim_failure_count.pop(target_id, None)
                self._state.last_verified_review_iteration = self.iteration
                console.print(f"  [green]{target_id} VERIFIED[/green]")
                # Auto-promote WH→ER if dependencies are satisfied
                if target_id.startswith("WH-"):
                    self._auto_promote(target_id)
                    # Update banner entry if promotion happened
                    er_id = f"ER-{target_id.split('-')[1]}"
                    if er_id in self.research_state.hypotheses:
                        self._state.pending_verified_results[-1]["claim"] = er_id
            elif verdict == Verdict.REFUTED:
                # Mark existing evidence as refuted (specific evidence was challenged)
                for ev in h.evidence:
                    ev.refuted = True
                h.refuted_count += 1
                count = self._state.claim_failure_count.get(target_id, 0) + 1
                self._state.claim_failure_count[target_id] = count
                self._state.pending_compute_verdicts.append(
                    {
                        "verdict": verdict,
                        "claim": target_id,
                        "attempt": count,
                        "notes": h.review.summary or "",
                        "details": h.review.details or "",
                        "task_id": task.task_id,
                    }
                )
                detail = (
                    h.review.summary[:120].replace("\n", " ")
                    if h.review.summary
                    else ""
                )
                console.print(f"  [red]{target_id} REFUTED[/red] — {detail}")
            else:
                # INCONCLUSIVE: keep existing evidence (not wrong, just insufficient)
                count = self._state.claim_failure_count.get(target_id, 0) + 1
                self._state.claim_failure_count[target_id] = count
                self._state.pending_compute_verdicts.append(
                    {
                        "verdict": verdict,
                        "claim": target_id,
                        "attempt": count,
                        "notes": h.review.summary or "",
                        "details": h.review.details or "",
                        "task_id": task.task_id,
                    }
                )
                detail = (
                    h.review.summary[:120].replace("\n", " ")
                    if h.review.summary
                    else ""
                )
                console.print(f"  [yellow]{target_id} INCONCLUSIVE[/yellow] — {detail}")

    def _update_research_iteration(self):
        """Update the iteration field in research state."""
        self.research_state.iteration = self.iteration

    def _auto_expire_critiques(self) -> None:
        """Auto-expire MEDIUM/LOW critiques older than auto_expire_iterations."""
        from .research_state import CritiqueStatus, Severity

        ttl = self.config.auto_expire_iterations
        if ttl <= 0:
            return
        expired = []
        for crit in self.research_state.critiques.values():
            if crit.status != CritiqueStatus.ACTIVE:
                continue
            if crit.severity == Severity.HIGH:
                continue
            if self.iteration - crit.iteration_filed >= ttl:
                crit.status = CritiqueStatus.RESOLVED
                crit.resolution_type = "expired"
                crit.resolution = f"Auto-expired after {self.iteration - crit.iteration_filed} iterations without resolution"
                crit.iteration_resolved = self.iteration
                expired.append(crit.id)
        if expired:
            log_scaffold_event(
                self.workspace.root,
                self.iteration,
                CC.LOOP_CONTROL,
                "auto_expire_critiques",
                f"ids={','.join(expired)}",
            )
            console.print(
                f"[dim]Auto-expired {len(expired)} critique(s): {', '.join(expired)}[/dim]"
            )

    def _set_research_status(self, status: str):
        """Update the status field in research state."""
        self.research_state.status = status

    def _render_files_for_git(self):
        """Render markdown files from ResearchState for git snapshots and verify.py."""
        self.workspace.write_file(
            "RESEARCH_STATE.md", render_research_state_md(self.research_state)
        )
        self.workspace.write_file(
            "EVIDENCE_LOG.md", render_evidence_log_md(self.research_state)
        )
        self.workspace.write_file(
            "CRITIQUE_LOG.md", render_critique_log_md(self.research_state)
        )

    def _run_formatter(self, answer_ers: list[str] | None = None) -> str | None:
        """Run the formatter agent to produce ANSWER.md.

        Returns ``None`` on success, or a rejection reason string if the
        formatter could not produce a valid answer matching the template.
        """
        console.print("[cyan]Formatter[/cyan] producing ANSWER.md...")
        fmt_task = Task(
            task_id=f"FORMAT-{self.iteration:03d}",
            task_type=TaskType.FORMAT,
            assigned_to="formatter",
            iteration=self.iteration,
            answer_ers=answer_ers or [],
        )
        self.formatter.research_state = self.research_state
        result = self.formatter.run(fmt_task, self.iteration)
        self._print_call_summary(result)

        rejection = self.formatter.rejection_reason
        if rejection:
            console.print(f"[yellow]Formatter rejected: {rejection}[/yellow]")
            log_scaffold_event(
                self.workspace.root,
                self.iteration,
                CC.LOOP_CONTROL,
                "formatter_rejection",
                rejection[:200],
            )
            return rejection

        self._render_files_for_git()
        self.workspace.git_commit(f"Iteration {self.iteration}: formatter - ANSWER.md")
        return None

    def _force_abandon_working_hypotheses(self):
        """Circuit breaker: auto-abandon all remaining WHs after repeated termination blocks."""
        from .research_state import FailedApproach, HypothesisStatus

        working = self.research_state.working_hypotheses()
        if not working:
            return
        ids = []
        for h in working:
            h.status = HypothesisStatus.ABANDONED
            h.iteration_modified = self.iteration
            self.research_state.failed_approaches.append(
                FailedApproach(
                    description=f"Auto-abandoned {h.id} — {h.statement or h.id}",
                    reason=(
                        f"Scaffolding circuit breaker: orchestrator attempted to terminate "
                        f"{self.config.max_termination_retries} times but {h.id} blocked it. "
                        f"No review-kind VERIFIED result was obtained."
                    ),
                    iteration=self.iteration,
                )
            )
            ids.append(h.id)
        log_scaffold_event(
            self.workspace.root,
            self.iteration,
            CC.LOOP_CONTROL,
            "termination_circuit_breaker",
            f"auto-abandoned {', '.join(ids)} after "
            f"{self._state.consecutive_termination_blocks} consecutive blocks",
        )
        console.print(
            f"[yellow]Circuit breaker: auto-abandoned {', '.join(ids)} "
            f"after {self._state.consecutive_termination_blocks} consecutive "
            f"termination blocks[/yellow]"
        )
        self._state.consecutive_termination_blocks = 0

    def _check_status_field(self) -> bool:
        """Check termination conditions beyond max_iterations."""
        if self.research_state.status in (
            "completed",
            "abandoned",
            "partially_complete",
        ):
            log_scaffold_event(
                self.workspace.root,
                self.iteration,
                CC.LOOP_CONTROL,
                "status_field_exit",
                f"status={self.research_state.status}",
            )
            return True
        return False

    def _sync_research_state(self):
        """Save ResearchState to workspace as JSON.

        The state is now authoritative — no rebuild from markdown needed.
        """
        normalize_references(self.research_state)
        self.research_state.save(self.workspace.root)

    def _update_metrics(self):
        """Write current metrics to METRICS.md."""
        md = self.metrics.to_markdown()
        self.workspace.write_file("METRICS.md", md)

    def _print_task(self, task: Task):
        """Print task summary to console."""
        print_task(task)

    def _on_compute_round(
        self,
        round_num,
        stop_reason,
        tool_calls,
        total_input,
        total_output,
        round_input,
        round_output,
        round_duration,
        round_reasoning=0,
        round_answer=0,
    ):
        """Progress callback for agent tool-use rounds."""
        on_round_progress(
            round_num,
            stop_reason,
            tool_calls,
            total_input,
            total_output,
            round_input,
            round_output,
            round_duration,
            round_reasoning,
            round_answer,
        )

    # Alias so orchestrator/critic use the same callback
    _on_agent_round = _on_compute_round

    def _print_call_summary(self, result):
        """Print a one-line timing/token summary for one-shot LLM calls."""
        print_call_summary(result)

    def _final_report(self):
        """Flush metrics and print final summary."""
        self._update_metrics()
        self._render_files_for_git()
        self.workspace.git_commit(f"Final metrics flush (iteration {self.iteration})")
        print_final_report(
            self.iteration,
            self.metrics,
            self.config,
            self.workspace.root,
        )
