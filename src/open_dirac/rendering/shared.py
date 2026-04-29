"""Shared rendering primitives used across multiple agent context builders.

These helpers turn `ResearchState` fragments into XML-delimited strings.
Per-agent context builders live next to each agent in
`src/open_dirac/agents/<name>/context.py`.
"""

from __future__ import annotations

from ..research_state import (
    FailedApproach,
    ResearchState,
    SanityCheck,
)


def _dedup_failed_approaches(approaches: list[FailedApproach]) -> list[FailedApproach]:
    """Keep only the latest FailedApproach per primary entity, preserving order."""
    best: dict[str, FailedApproach] = {}
    for fa in approaches:
        key = fa.related_entities[0] if fa.related_entities else fa.description
        if key not in best or fa.iteration >= best[key].iteration:
            best[key] = fa
    # Preserve first-seen order of keys
    seen: set[str] = set()
    result: list[FailedApproach] = []
    for fa in approaches:
        key = fa.related_entities[0] if fa.related_entities else fa.description
        if key not in seen and best[key] is fa:
            seen.add(key)
            result.append(fa)
    return result


def _render_sanity_checks(checks: list[SanityCheck], tag: str = "sanity-checks") -> str:
    """Render structured sanity checks as an XML-tagged block."""
    lines: list[str] = []
    for sc in checks:
        line = f"- [{sc.id}] {sc.predicate}"
        if sc.rationale:
            line += f"\n  Rationale: {sc.rationale}"
        lines.append(line)
    return f"<{tag}>\n" + "\n".join(lines) + f"\n</{tag}>"


def render_background_survey_xml(state: ResearchState) -> str:
    """Render survey data as XML sub-tags for agent context.

    Returns inner content (background, key-insights, known-methods, known-pitfalls)
    without an outer wrapper — callers wrap in <background-survey> or
    <current-background-survey> as appropriate.
    """
    parts: list[str] = []
    if state.survey_background:
        parts.append(f"<background>\n{state.survey_background}\n</background>")
    if state.key_insights:
        parts.append(f"<key-insights>\n{state.key_insights}\n</key-insights>")
    if state.survey_methods:
        parts.append(f"<known-methods>\n{state.survey_methods}\n</known-methods>")
    if state.known_pitfalls:
        parts.append(f"<known-pitfalls>\n{state.known_pitfalls}\n</known-pitfalls>")
    if state.expected_answer_structure:
        parts.append(
            f"<expected-answer-structure>\n{state.expected_answer_structure}\n</expected-answer-structure>"
        )
    return "\n".join(parts)


def _problem_guidelines() -> str:
    """Return the <problem-guidelines> block shared by all agent contexts."""
    return (
        "<problem-guidelines>\n"
        "- The problem statement is correct and well-posed. Do not question "
        "whether the problem contains errors or is ill-defined.\n"
        "- The answer template suggests a format, but do not infer that the "
        "final answer must depend on every parameter appearing in the template. "
        "A parameter's presence in the template does not guarantee it survives "
        "in the final expression.\n"
        "- The answer must be valid for every value of the declared domain of "
        "each parameter in the docstring. If a general formula is undefined at a boundary point "
        "that belongs to the domain (e.g. 0/0, log(0)), use a Sympy "
        "construct to return the correct limiting value there.\n"
        "</problem-guidelines>"
    )


def render_research_context_xml(state: ResearchState) -> str:
    """Render <research-context> wrapper: problem-statement + answer-template."""
    parts = [f"<problem-statement>\n{state.problem_statement}\n</problem-statement>"]
    if state.answer_template:
        parts.append(f"<answer-template>\n{state.answer_template}\n</answer-template>")
    parts.append(_problem_guidelines())
    return "<research-context>\n" + "\n".join(parts) + "\n</research-context>"
