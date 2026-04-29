"""Diagnosis pass: LLM-driven audit of a completed research run.

Builds a structured prompt from :class:`WorkspaceContents`, calls the LLM,
parses the XML-tagged response, renders it to the console, and appends the
result to ``VERIFICATION.md``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.table import Table

from ..config import Config
from ..console import console
from ..llm import LLMResponse, _call_provider_with_retry, _get_provider
from ..utils.markdown import parse_frontmatter, render_frontmatter
from .event_summary import summarize_event_log
from .formal_eval import FormalEvalResult
from .workspace import RerunResult, WorkspaceContents

PROMPTS_DIR = Path(__file__).parent

CLASSIFICATION_COLORS = {
    "CAUGHT": "green",
    "UNCAUGHT": "red",
    "PARTIAL": "yellow",
}


@dataclass
class DiagnosisEvent:
    """A single event in a failure chain or correction chain."""

    event_id: str
    chain_type: str  # "correction_chain" / "failure_chain"
    classification: str  # CAUGHT / UNCAUGHT / PARTIAL
    agents_involved: list[str] = field(default_factory=list)
    iterations: str = ""
    description: str = ""
    root_cause: str = ""
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class DiagnosisResult:
    """Result of the unified diagnosis pass."""

    formal_outcome: str = ""  # CORRECT / INCORRECT / INCONCLUSIVE / SKIPPED
    diagnosis_mode: str = ""  # success_analysis / failure_analysis
    summary: str = ""
    events: list[DiagnosisEvent] = field(default_factory=list)
    weakest_link: str = ""
    recommendations: list[str] = field(default_factory=list)
    raw_response: str = ""
    parse_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_diagnosis_llm(system: str, user_content: str, config: Config) -> LLMResponse:
    """Call the configured LLM for the diagnosis pass via the provider layer.

    One-shot call (no tools, no agent loop), retried on transient errors. The
    provider handles its own streaming/batching internally.
    """
    provider = _get_provider(config)
    start = time.time()
    resp = _call_provider_with_retry(
        provider,
        config,
        workspace_dir=config.workspace_dir,
        model=config.model_id,
        max_tokens=config.max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    duration = time.time() - start
    return LLMResponse(
        text=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        stop_reason=resp.stop_reason,
        duration=duration,
        reasoning_tokens=resp.reasoning_tokens,
        answer_tokens=resp.answer_tokens,
        reasoning_content=resp.reasoning_content,
    )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _formal_eval_outcome(formal_eval: FormalEvalResult | None) -> str:
    """Return a string label for the formal evaluation outcome."""
    if formal_eval is None or formal_eval.skipped:
        return "SKIPPED"
    if formal_eval.correct is True:
        return "CORRECT"
    if formal_eval.correct is False:
        return "INCORRECT"
    return "INCONCLUSIVE"


def build_diagnosis_prompt(
    contents: WorkspaceContents,
    formal_eval: FormalEvalResult | None = None,
    rerun_results: list[RerunResult] | None = None,
    known_answer: str | None = None,
    reference_content: str | None = None,
) -> tuple[str, str]:
    """Assemble the system prompt and user content for the unified diagnosis LLM call."""
    system = (PROMPTS_DIR / "diagnosis.md").read_text()

    sections = []

    outcome = _formal_eval_outcome(formal_eval)
    if outcome == "CORRECT":
        sections.append(
            "## Formal Answer Evaluation: CORRECT\n\n"
            "An automated symbolic/numerical comparison confirmed the final answer is **CORRECT**."
        )
        if formal_eval.method:
            sections.append(f"(method: {formal_eval.method})")
        sections.append(
            "\nThe system arrived at the right answer. Focus your analysis on "
            "**correction chains**: errors that were made during the run and how the "
            "system caught and corrected them.\n"
        )
    elif outcome == "INCORRECT":
        sections.append(
            "## Formal Answer Evaluation: INCORRECT\n\n"
            "An automated symbolic/numerical comparison confirmed the final answer is **INCORRECT**."
        )
        if formal_eval.method:
            sections.append(f"(method: {formal_eval.method})")
        if formal_eval.details:
            sections.append(f"Details: {formal_eval.details}")
        sections.append(
            "\nThe system failed to reach the right answer. Focus your analysis on "
            "**failure chains**: root-cause analysis of where reasoning went wrong and "
            "why self-correction mechanisms failed to catch it.\n"
        )
    elif outcome == "INCONCLUSIVE":
        sections.append(
            "## Formal Answer Evaluation: INCONCLUSIVE\n\n"
            "The automated comparison could not determine correctness."
        )
        if formal_eval.error:
            sections.append(f"Error: {formal_eval.error}")
        sections.append(
            "\nAnalyze both the scientific validity of the result and the process quality.\n"
        )
    else:
        sections.append(
            "## Formal Answer Evaluation: SKIPPED\n\n"
            "No ground truth is available for automated comparison."
        )
        if formal_eval and formal_eval.skip_reason:
            sections.append(f"Reason: {formal_eval.skip_reason}")
        sections.append(
            "\nAnalyze both the scientific plausibility of the result and the process quality.\n"
        )

    if known_answer:
        sections.append(
            f"## Known Answer\n\nThe expected answer is: **{known_answer}**\n"
        )

    if reference_content:
        sections.append(
            "## Reference Document\n\n"
            "The following reference document contains the ground-truth answer "
            "and a description of what a typical successful run looks like.\n\n"
            f"```markdown\n{reference_content}\n```\n"
        )

    if contents.terminated_cleanly:
        sections.append("## Termination Status\nThe research run terminated cleanly.\n")
    else:
        sections.append(
            "## Termination Status\n⚠ The research run did NOT terminate cleanly. "
            "Results may be incomplete.\n"
        )

    for label, text in [
        ("RESEARCH_STATE.md", contents.research_state),
        ("EVIDENCE_LOG.md", contents.evidence_log),
        ("CRITIQUE_LOG.md", contents.critique_log),
        ("CURRENT_TASK.md", contents.current_task),
    ]:
        if text:
            sections.append(f"## {label}\n\n```markdown\n{text}\n```\n")
        else:
            sections.append(f"## {label}\n\n(File not found or empty.)\n")

    if contents.background_survey:
        sections.append(
            f"## Background Survey (Surveyor Output)\n\n```markdown\n{contents.background_survey}\n```\n"
        )

    if contents.metrics_md:
        sections.append(f"## METRICS.md\n\n```markdown\n{contents.metrics_md}\n```\n")

    if contents.git_log:
        sections.append(f"## Git Log\n\n```\n{contents.git_log}\n```\n")

    if contents.event_log:
        summary = summarize_event_log(contents.event_log)
        if summary:
            sections.append(f"## Event Log Summary\n\n{summary}\n")

    if rerun_results:
        sections.append("## Computation Re-run Results\n")
        for rr in rerun_results:
            name = Path(rr.script_path).name
            ex = rr.execution
            if ex is None:
                sections.append(f"### {name}\n(Not executed.)\n")
                continue
            status = (
                "TIMED OUT"
                if ex.timed_out
                else (
                    "SUCCESS" if ex.returncode == 0 else f"FAILED (rc={ex.returncode})"
                )
            )
            sections.append(f"### {name} — {status}\n")
            if ex.stdout.strip():
                sections.append(f"**stdout:**\n```\n{ex.stdout.strip()}\n```\n")
            if ex.stderr.strip():
                sections.append(f"**stderr:**\n```\n{ex.stderr.strip()}\n```\n")

    user_content = "\n".join(sections)
    return system, user_content


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_tag(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag>. Returns '' if not found."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


_EVENT_HEADER_RE = re.compile(
    r"(EVENT-\d+)\s+\[(CAUGHT|UNCAUGHT|PARTIAL)]\s+\(([^)]*)\)\s*\n(.*?)(?=EVENT-\d+|\Z)",
    re.DOTALL,
)


def _parse_chain_events(chains_text: str) -> list[DiagnosisEvent]:
    """Extract per-event records from the <chains> block."""
    events: list[DiagnosisEvent] = []
    for m in _EVENT_HEADER_RE.finditer(chains_text):
        body = m.group(4).strip()

        agents: list[str] = []
        agents_match = re.search(r"Agents?:\s*(.+)", body)
        if agents_match:
            agents = [a.strip() for a in agents_match.group(1).split(",")]

        root_cause = ""
        rc_match = re.search(r"Root cause:\s*(.+)", body)
        if rc_match:
            root_cause = rc_match.group(1).strip()

        evidence_ids: list[str] = []
        ev_match = re.search(r"Evidence:\s*(.+)", body)
        if ev_match:
            evidence_ids = [e.strip() for e in ev_match.group(1).split(",")]

        description = body
        for pattern in [r"\nAgents?:.*", r"\nRoot cause:.*", r"\nEvidence:.*"]:
            description = re.sub(pattern, "", description, flags=re.DOTALL)
        description = description.strip()

        classification = m.group(2)
        chain_type = (
            "correction_chain" if classification == "CAUGHT" else "failure_chain"
        )

        events.append(
            DiagnosisEvent(
                event_id=m.group(1),
                chain_type=chain_type,
                classification=classification,
                agents_involved=agents,
                iterations=m.group(3).strip(),
                description=description,
                root_cause=root_cause,
                evidence_ids=evidence_ids,
            )
        )
    return events


def parse_diagnosis(
    response_text: str, formal_eval: FormalEvalResult | None = None
) -> DiagnosisResult:
    """Parse XML-tagged sections from the diagnosis LLM response."""
    outcome = _formal_eval_outcome(formal_eval)
    mode = "success_analysis" if outcome == "CORRECT" else "failure_analysis"
    result = DiagnosisResult(
        formal_outcome=outcome,
        diagnosis_mode=mode,
        raw_response=response_text,
    )

    summary = _extract_tag(response_text, "diagnosis_summary")
    if summary:
        result.summary = summary
    else:
        result.parse_warnings.append("No <diagnosis_summary> tag found")

    chains_text = _extract_tag(response_text, "chains")
    if chains_text:
        result.events = _parse_chain_events(chains_text)
    else:
        result.parse_warnings.append("No <chains> tag found")

    weakest = _extract_tag(response_text, "weakest_link")
    if weakest:
        result.weakest_link = weakest
    else:
        result.parse_warnings.append("No <weakest_link> tag found")

    recs_text = _extract_tag(response_text, "recommendations")
    if recs_text:
        for line in recs_text.splitlines():
            line = line.strip().lstrip("-").strip()
            if line:
                result.recommendations.append(line)
    else:
        result.parse_warnings.append("No <recommendations> tag found")

    return result


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------


def render_diagnosis(result: DiagnosisResult) -> None:
    """Print the diagnosis result to the console using Rich."""
    if result.diagnosis_mode == "success_analysis":
        console.print("\n[bold green]--- Diagnosis: Success Analysis ---[/]")
    else:
        console.print("\n[bold red]--- Diagnosis: Failure Analysis ---[/]")
    console.print()

    if result.summary:
        console.print(f"[bold]Summary:[/] {result.summary}")
        console.print()

    if result.events:
        table = Table(title="Error/Correction Chains")
        table.add_column("Event", style="cyan")
        table.add_column("Status")
        table.add_column("Iterations")
        table.add_column("Agents")
        table.add_column("Description", max_width=50)
        for ev in result.events:
            color = CLASSIFICATION_COLORS.get(ev.classification, "white")
            table.add_row(
                ev.event_id,
                f"[{color}]{ev.classification}[/]",
                ev.iterations,
                ", ".join(ev.agents_involved) if ev.agents_involved else "",
                ev.description[:150],
            )
        console.print(table)
        console.print()

    if result.weakest_link:
        console.print(f"[bold]Weakest link:[/] {result.weakest_link}")
        console.print()

    if result.recommendations:
        console.print("[bold]Recommendations:[/]")
        for r in result.recommendations:
            console.print(f"  - {r}")
        console.print()

    if result.parse_warnings:
        console.print("[dim]Parse warnings:[/]")
        for w in result.parse_warnings:
            console.print(f"  [dim]- {w}[/]")
        console.print()


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_diagnosis_report(result: DiagnosisResult, workspace_dir: str) -> None:
    """Append diagnosis sections to VERIFICATION.md (created by engine's formal eval)."""
    report_path = Path(workspace_dir) / "VERIFICATION.md"

    if report_path.exists():
        existing = report_path.read_text()
    else:
        existing = "---\n---\n\n# Verification Report\n"

    # Round-trip the frontmatter through YAML so diagnosis_mode is added safely.
    # Previously this was a regex patch that silently lost the block on any mismatch.
    fm, body = parse_frontmatter(existing)
    fm["diagnosis_mode"] = result.diagnosis_mode
    existing = render_frontmatter(fm, body.lstrip("\n"))

    sections = ["\n---\n"]

    if result.diagnosis_mode == "success_analysis":
        sections.append("# Diagnosis: Success Analysis\n")
    else:
        sections.append("# Diagnosis: Failure Analysis\n")

    if result.summary:
        sections.append(f"## Summary\n\n{result.summary}\n")

    if result.events:
        sections.append("## Error/Correction Chains\n")
        for ev in result.events:
            sections.append(
                f"### {ev.event_id} [{ev.classification}] ({ev.iterations})\n"
            )
            if ev.agents_involved:
                sections.append(f"**Agents:** {', '.join(ev.agents_involved)}\n")
            if ev.description:
                sections.append(f"{ev.description}\n")
            if ev.root_cause:
                sections.append(f"**Root cause:** {ev.root_cause}\n")
            if ev.evidence_ids:
                sections.append(f"**Evidence:** {', '.join(ev.evidence_ids)}\n")

    if result.weakest_link:
        sections.append(f"## Weakest Link\n\n{result.weakest_link}\n")

    if result.recommendations:
        sections.append("## Recommendations\n")
        for r in result.recommendations:
            sections.append(f"- {r}")
        sections.append("")

    report_path.write_text(existing + "\n".join(sections))
    console.print(f"[green]Diagnosis appended to {report_path}[/]")
