"""Formal research state tracking.

Structured representation of hypotheses, evidence, critiques, and their
relationships.  Agents mutate state via tools, Markdown rendered from it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HypothesisStatus(StrEnum):
    WORKING = "working"
    ESTABLISHED = "established"
    REFUTED = "refuted"
    ABANDONED = "abandoned"


class Verdict(StrEnum):
    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class Severity(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CritiqueStatus(StrEnum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    WITHDRAWN = "withdrawn"


class RQStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


# ---------------------------------------------------------------------------
# Entity dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """Evidence produced by a researcher or computer agent."""

    id: str = ""  # EV-NNN, assigned by _store_evidence
    type: str = ""  # "research" or "compute"
    reasoning: str = ""  # Researcher's analytical work
    approach: str = ""  # Computer's document_approach output
    scripts: list[str] = field(default_factory=list)
    script_purposes: dict[str, str] = field(default_factory=dict)
    output: str = ""  # Code execution output summary
    method: str = ""
    result: str = ""
    description: str = ""  # What was computed (computer submit_result)
    notes: str = ""  # Additional notes (computer submit_result)
    confidence: str = ""  # exact/approximate/partial
    summary: str = ""  # One-sentence summary for banners
    iteration: int | None = None
    derivation_file: str = ""  # Filename in derivations/ (researcher only)
    refuted: bool = False  # Marked True when review verdict is REFUTED

    def __post_init__(self):
        # LLMs may return non-string values (e.g. int) for these fields
        if not isinstance(self.result, str):
            self.result = str(self.result)
        if not isinstance(self.summary, str):
            self.summary = str(self.summary)


@dataclass
class ReviewResult:
    """Review result produced by the reviewer agent."""

    verdict: str = ""  # VERIFIED/REFUTED/INCONCLUSIVE
    summary: str = ""
    details: str = ""
    iteration: int | None = None


@dataclass
class Hypothesis:
    id: str
    statement: str = ""
    status: HypothesisStatus = HypothesisStatus.WORKING
    derivation: str = ""
    critiques: list[str] = field(default_factory=list)
    iteration_created: int = 0
    iteration_modified: int = 0
    depends_on: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    review: ReviewResult | None = None
    refuted_count: int = 0


@dataclass
class Critique:
    id: str
    targets: list[str] = field(default_factory=list)
    severity: Severity = Severity.MEDIUM
    argument: str = ""
    status: CritiqueStatus = CritiqueStatus.ACTIVE
    resolution: str = ""
    resolution_type: str = ""  # "dismissed" or "accepted"
    target_type: str = ""  # "er", "strategy", "coordination" (from critic)
    iteration_filed: int = 0
    iteration_resolved: int | None = None
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class SanityCheck:
    """A testable constraint for candidate answers."""

    id: str  # SC-001, SC-002, ...
    predicate: str  # Testable condition (pass/fail statement)
    rationale: str = ""  # Why this check should hold


@dataclass
class FailedApproach:
    description: str = ""
    reason: str = ""
    related_entities: list[str] = field(default_factory=list)
    iteration: int = 0
    derivation_excerpt: str = ""


@dataclass
class ResearchQuestion:
    """Open-ended research question (not a falsifiable claim)."""

    id: str  # RQ-NNN
    question: str = ""
    context: str = ""  # why this question matters
    resolved_to: list[str] = field(default_factory=list)  # WH-NNN IDs
    status: RQStatus = RQStatus.OPEN
    iteration_created: int = 0
    iteration_resolved: int | None = None
    resolution_reason: str = ""  # why / how it was resolved
    evidence: list[Evidence] = field(
        default_factory=list
    )  # evidence from researcher/computer


# ---------------------------------------------------------------------------
# ResearchState
# ---------------------------------------------------------------------------

STATE_FILENAME = "RESEARCH_GRAPH.json"

_ER_WH_SECTION_RE = re.compile(r"^## ((?:ER|WH)-\d+)\s*[-—:]?\s*(.*)", re.MULTILINE)
_SECTION_END_RE = re.compile(r"^(?:##? )", re.MULTILINE)


@dataclass
class ResearchState:
    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    critiques: dict[str, Critique] = field(default_factory=dict)
    research_questions: dict[str, ResearchQuestion] = field(default_factory=dict)
    failed_approaches: list[FailedApproach] = field(default_factory=list)
    critic_clean_reviews: list[dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    problem_statement: str = ""
    answer_template: str = ""
    conventions: str = ""
    strategy: str = ""
    research_notes: list[dict] = field(default_factory=list)
    status: str = "in_progress"
    title: str = ""
    sanity_checks: list[SanityCheck] = field(
        default_factory=list
    )  # seeded by surveyor, editable by planner
    # Survey-produced fields (seeded by surveyor, immutable after)
    survey_background: str = ""  # §1: background context
    key_insights: str = ""  # §2: key insights
    survey_methods: str = ""  # §3: known methods and techniques
    known_pitfalls: str = ""  # §4: known pitfalls
    expected_answer_structure: str = ""  # §7: expected form/complexity of the answer
    problem_summary: str = ""  # §8: one-sentence problem summary

    # --- Query methods ---

    def has_verified_evidence(self, hypothesis_id: str) -> bool:
        """True if hypothesis has a VERIFIED review result."""
        h = self.hypotheses.get(hypothesis_id)
        if not h or not h.review:
            return False
        return h.review.verdict == Verdict.VERIFIED

    def hypotheses_with_evidence(self) -> list[Hypothesis]:
        """Hypotheses that have evidence attached."""
        return [h for h in self.hypotheses.values() if h.evidence]

    def active_critiques_for(self, target_id: str) -> list[Critique]:
        """Active critiques mentioning *target_id*."""
        return [
            c
            for c in self.critiques.values()
            if target_id in c.targets and c.status == CritiqueStatus.ACTIVE
        ]

    def established_hypotheses(self) -> list[Hypothesis]:
        return [
            h
            for h in self.hypotheses.values()
            if h.status == HypothesisStatus.ESTABLISHED
        ]

    def working_hypotheses(self) -> list[Hypothesis]:
        return [
            h for h in self.hypotheses.values() if h.status == HypothesisStatus.WORKING
        ]

    def abandoned_hypotheses(self) -> list[Hypothesis]:
        return [
            h
            for h in self.hypotheses.values()
            if h.status == HypothesisStatus.ABANDONED
        ]

    def failures_for_hypothesis(self, hypothesis_id: str) -> list[FailedApproach]:
        """Failed approaches mentioning *hypothesis_id*."""
        return [
            fa
            for fa in self.failed_approaches
            if hypothesis_id in fa.description
            or hypothesis_id in " ".join(fa.related_entities)
        ]

    def open_research_questions(self) -> list[ResearchQuestion]:
        """All open research questions."""
        return [
            rq for rq in self.research_questions.values() if rq.status == RQStatus.OPEN
        ]

    def next_entity_num(self) -> int:
        """Max existing entity number across hypotheses and RQs, + 1.

        RQ, WH, and ER share a single counter so that an entity keeps the
        same number through its lifecycle (RQ-003 → WH-003 → ER-003).
        """
        nums = []
        for hid in self.hypotheses:
            parts = hid.split("-")
            if len(parts) == 2:
                try:
                    nums.append(int(parts[1]))
                except ValueError:
                    pass
        for rqid in self.research_questions:
            parts = rqid.split("-")
            if len(parts) == 2 and parts[0] == "RQ":
                try:
                    nums.append(int(parts[1]))
                except ValueError:
                    pass
        return max(nums, default=0) + 1

    def next_rq_num(self) -> int:
        """Max existing RQ number + 1 (delegates to unified counter)."""
        return self.next_entity_num()

    def next_hypothesis_num(self) -> int:
        """Max existing hypothesis number + 1 (delegates to unified counter)."""
        return self.next_entity_num()

    def unestablished_dependencies(self, hypothesis_id: str) -> list[str]:
        """Return dependency IDs that are not yet ESTABLISHED."""
        if hypothesis_id not in self.hypotheses:
            return []
        deps = self.hypotheses[hypothesis_id].depends_on
        return [
            d
            for d in deps
            if d not in self.hypotheses
            or self.hypotheses[d].status != HypothesisStatus.ESTABLISHED
        ]

    def next_critique_num(self) -> int:
        """Max existing CRIT number + 1."""
        nums = []
        for cid in self.critiques:
            parts = cid.split("-")
            if len(parts) == 2 and parts[0] == "CRIT":
                try:
                    nums.append(int(parts[1]))
                except ValueError:
                    pass
        return max(nums, default=0) + 1

    def next_evidence_num(self) -> int:
        """Max existing EV number across all entities + 1."""
        nums: list[int] = []
        for collection in (
            self.hypotheses.values(),
            self.research_questions.values(),
            self.critiques.values(),
        ):
            for entity in collection:
                for ev in entity.evidence:
                    if ev.id and ev.id.startswith("EV-"):
                        try:
                            nums.append(int(ev.id.split("-")[1]))
                        except ValueError:
                            pass
        return max(nums, default=0) + 1

    def next_sc_num(self) -> int:
        """Max existing SC number + 1."""
        nums = []
        for sc in self.sanity_checks:
            parts = sc.id.split("-")
            if len(parts) == 2 and parts[0] == "SC":
                try:
                    nums.append(int(parts[1]))
                except ValueError:
                    pass
        return max(nums, default=0) + 1

    # --- Serialization ---

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> ResearchState:
        data = json.loads(text)
        state = cls(
            iteration=data.get("iteration", 0),
            problem_statement=data.get("problem_statement", ""),
            answer_template=data.get("answer_template", ""),
            conventions=data.get("conventions", ""),
            strategy=data.get("strategy", ""),
            research_notes=data.get("research_notes", []),
            status=data.get("status", "in_progress"),
            title=data.get("title", ""),
        )

        def _parse_evidence_list(raw: list[dict] | None) -> list[Evidence]:
            if not raw:
                return []
            return [
                Evidence(
                    id=edata.get("id", ""),
                    type=edata.get("type", ""),
                    reasoning=edata.get("reasoning", ""),
                    approach=edata.get("approach", ""),
                    scripts=edata.get("scripts", []),
                    script_purposes=edata.get("script_purposes", {}),
                    output=edata.get("output", ""),
                    method=edata.get("method", ""),
                    result=edata.get("result", ""),
                    description=edata.get("description", ""),
                    notes=edata.get("notes", ""),
                    confidence=edata.get("confidence", ""),
                    summary=edata.get("summary", ""),
                    iteration=edata.get("iteration"),
                    derivation_file=edata.get("derivation_file", ""),
                    refuted=edata.get("refuted", False),
                )
                for edata in raw
            ]

        for hid, hdata in data.get("hypotheses", {}).items():
            evidence = _parse_evidence_list(hdata.get("evidence"))
            review = None
            vdata = hdata.get("review")
            if vdata:
                review = ReviewResult(
                    verdict=vdata.get("verdict", ""),
                    summary=vdata.get("summary", ""),
                    details=vdata.get("details", ""),
                    iteration=vdata.get("iteration"),
                )
            state.hypotheses[hid] = Hypothesis(
                id=hdata["id"],
                statement=hdata.get("statement", ""),
                status=HypothesisStatus(hdata.get("status", "working")),
                derivation=hdata.get("derivation", ""),
                critiques=hdata.get("critiques", []),
                iteration_created=hdata.get("iteration_created", 0),
                iteration_modified=hdata.get("iteration_modified", 0),
                depends_on=hdata.get("depends_on", []),
                evidence=evidence,
                review=review,
                refuted_count=hdata.get("refuted_count", 0),
            )
        for crid, crdata in data.get("critiques", {}).items():
            crit_evidence = _parse_evidence_list(crdata.get("evidence"))
            state.critiques[crid] = Critique(
                id=crdata["id"],
                targets=crdata.get("targets", []),
                severity=Severity(crdata.get("severity", "MEDIUM")),
                argument=crdata.get("argument", ""),
                status=CritiqueStatus(crdata.get("status", "active")),
                resolution=crdata.get("resolution", ""),
                resolution_type=crdata.get("resolution_type", ""),
                target_type=crdata.get("target_type", ""),
                iteration_filed=crdata.get("iteration_filed", 0),
                iteration_resolved=crdata.get("iteration_resolved"),
                evidence=crit_evidence,
            )
        for rqid, rqdata in data.get("research_questions", {}).items():
            rq_evidence = _parse_evidence_list(rqdata.get("evidence"))
            state.research_questions[rqid] = ResearchQuestion(
                id=rqdata["id"],
                question=rqdata.get("question", ""),
                context=rqdata.get("context", ""),
                resolved_to=rqdata.get("resolved_to", []),
                status=RQStatus(rqdata.get("status", "open")),
                iteration_created=rqdata.get("iteration_created", 0),
                iteration_resolved=rqdata.get("iteration_resolved"),
                resolution_reason=rqdata.get("resolution_reason", ""),
                evidence=rq_evidence,
            )
        for fdata in data.get("failed_approaches", []):
            state.failed_approaches.append(
                FailedApproach(
                    description=fdata.get("description", ""),
                    reason=fdata.get("reason", ""),
                    related_entities=fdata.get("related_entities", []),
                    iteration=fdata.get("iteration", 0),
                    derivation_excerpt=fdata.get("derivation_excerpt", ""),
                )
            )
        state.critic_clean_reviews = data.get("critic_clean_reviews", [])
        for c in data.get("sanity_checks", []):
            state.sanity_checks.append(
                SanityCheck(
                    id=c["id"],
                    predicate=c["predicate"],
                    rationale=c.get("rationale", ""),
                )
            )
        state.survey_background = data.get("survey_background", "")
        state.key_insights = data.get("key_insights", "")
        state.survey_methods = data.get("survey_methods", "")
        state.known_pitfalls = data.get("known_pitfalls", "")
        state.expected_answer_structure = data.get("expected_answer_structure", "")
        state.problem_summary = data.get("problem_summary", "")
        return state

    def save(self, workspace_root: Path) -> None:
        path = workspace_root / STATE_FILENAME
        path.write_text(self.to_json())

    @classmethod
    def load(cls, workspace_root: Path) -> ResearchState:
        path = workspace_root / STATE_FILENAME
        if path.exists():
            return cls.from_json(path.read_text())
        return cls()


# ---------------------------------------------------------------------------
# Markdown parsing helpers (used by tests for fixture setup)
# ---------------------------------------------------------------------------

_H1_SECTION_RE = re.compile(r"^# (.+)", re.MULTILINE)


def _extract_h1_section(body: str, heading: str) -> str:
    """Extract the content of an H1 section by heading name."""
    pattern = re.compile(rf"^# {re.escape(heading)}\s*$", re.MULTILINE)
    m = pattern.search(body)
    if not m:
        return ""
    start = m.end()
    # Find next H1
    rest = body[start:]
    next_h1 = _H1_SECTION_RE.search(rest)
    end = start + next_h1.start() if next_h1 else len(body)
    return body[start:end].strip()


def _extract_hypothesis_sections(body: str) -> list[tuple[str, str, str]]:
    """Extract (id, title, body_text) for each ## WH-NNN / ## ER-NNN section."""
    results = []
    matches = list(_ER_WH_SECTION_RE.finditer(body))
    for i, m in enumerate(matches):
        hid = m.group(1)
        title = m.group(2).strip().rstrip("*").strip()
        content_start = m.end()
        # Content extends until the next ## or # header, or end of string
        if i + 1 < len(matches):
            content_end = matches[i + 1].start()
        else:
            # Find next # or ## header after this section
            rest = body[content_start:]
            end_match = _SECTION_END_RE.search(rest)
            content_end = content_start + end_match.start() if end_match else len(body)
        section_body = body[content_start:content_end].strip()
        results.append((hid, title, section_body))
    return results
