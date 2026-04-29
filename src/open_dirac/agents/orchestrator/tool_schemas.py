"""Orchestrator tool schemas (OpenAI canonical format).

Declarative definitions of the state-mutation tools the orchestrator calls.
The executor that applies them lives in ``tools.py``.
"""

from __future__ import annotations


ORCHESTRATOR_TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "add_hypothesis",
            "description": (
                "Add a new Working Hypothesis from a Research Question and "
                "auto-dispatch the reviewer. "
                "Requires from_rq: every WH must originate from an RQ "
                "that has gathered evidence. "
                "ENDS YOUR TURN — the reviewer is auto-dispatched to check "
                "the new WH. Complete all other mutations before calling this. "
                "Returns the auto-assigned ID (e.g., WH-003)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "statement": {
                        "type": "string",
                        "description": "One-line title for the hypothesis.",
                    },
                    "derivation": {
                        "type": "string",
                        "description": "Full reasoning or derivation (Markdown).",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "IDs of hypotheses this claim depends on "
                            "(e.g. ['ER-001', 'WH-002']). Optional."
                        ),
                    },
                    "from_rq": {
                        "type": "string",
                        "description": (
                            "If this WH is the concrete result of a research "
                            "question, provide the RQ ID (e.g. 'RQ-003'). "
                            "The WH inherits the RQ's number and the RQ is "
                            "auto-resolved."
                        ),
                    },
                },
                "required": ["statement", "from_rq"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abandon_hypothesis",
            "description": (
                "Mark a hypothesis as a dead end. "
                "Removes it from the WH/ER section and records the reason under Dead Ends."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Hypothesis ID to abandon.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this approach was abandoned.",
                    },
                },
                "required": ["id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_convention",
            "description": (
                "Add new convention entries to the Conventions section. "
                "Only include NEW items — existing conventions are preserved automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": (
                            "New convention entries to append. "
                            "Do NOT repeat existing conventions."
                        ),
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_note",
            "description": (
                "Append a research note. Use for recording intermediate insights, "
                "observations, or decisions. Notes are append-only and visible to "
                "all agents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The note content.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_research_question",
            "description": (
                "Add an open-ended research question (RQ). "
                "Use for questions that need exploration before a concrete "
                "hypothesis can be formulated. Returns the auto-assigned ID (e.g., RQ-001)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The research question.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Why this question matters for the research.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abandon_research_question",
            "description": (
                "Mark a research question as a dead end. "
                "Use when the gathered evidence is inconclusive or the "
                "question turned out to be ill-posed. "
                "(RQs answered via add_hypothesis are auto-resolved.)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Research question ID, e.g. RQ-001.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this RQ is being abandoned.",
                    },
                },
                "required": ["id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_researcher",
            "description": (
                "Dispatch the researcher agent for pure reasoning, derivation, "
                "or analysis. No code execution. Can be called alongside "
                "mutation tools — all mutations are applied before dispatch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_claim": {
                        "type": "string",
                        "description": (
                            "The RQ or WH ID this task targets "
                            "(e.g. 'RQ-001', 'WH-003')."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Goal-focused task description. Lead with a single "
                            "sentence stating the deliverable and its scope. "
                            "Add critical constraints or pitfalls if needed. "
                            "Do NOT write step-by-step procedures — put method "
                            "suggestions in method_hints instead."
                        ),
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "background": {
                        "type": "string",
                        "description": (
                            "Task-specific context the agent cannot see on its own. "
                            "The agent already receives: problem summary, conventions, "
                            "and all established results. "
                            "Do NOT repeat those. Instead provide: (1) relevant parts of "
                            "the survey background or known methods, (2) strategic context — "
                            "why this task matters and how it fits the plan, (3) relevant "
                            "research notes or observations from prior iterations, "
                            "(4) inter-entity connections the agent should be aware of. "
                            "The agent has NO access to strategy, research notes, or "
                            "survey — relay the relevant parts here."
                        ),
                    },
                    "method_hints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Suggested methods or approaches for the agent to consider."
                        ),
                    },
                    "assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": ("Key assumptions the agent should work under."),
                    },
                    "relevant_results": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "References to established results or prior evidence "
                            "that are relevant to this task (e.g. 'ER-001', 'WH-003')."
                        ),
                    },
                    "recommended_sanity_checks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Sanity checks the agent should verify its result against. "
                            "Pick from the global sanity-checks list or write task-specific "
                            "ones (e.g. 'result must reduce to X in the Y limit')."
                        ),
                    },
                },
                "required": ["target_claim", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_computer",
            "description": (
                "Dispatch the computer agent for numerical, symbolic, or "
                "simulation work via Python/SymPy/NumPy/SciPy. Can be called "
                "alongside mutation tools — all mutations are applied before dispatch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_claim": {
                        "type": "string",
                        "description": (
                            "The RQ or WH ID this task targets "
                            "(e.g. 'RQ-001', 'WH-003')."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Goal-focused task description. Lead with a single "
                            "sentence stating the deliverable and its scope. "
                            "Add critical constraints or pitfalls if needed. "
                            "Do NOT write step-by-step procedures — put method "
                            "suggestions in method_hints instead."
                        ),
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "background": {
                        "type": "string",
                        "description": (
                            "Task-specific context the agent cannot see on its own. "
                            "The agent already receives: problem summary, conventions, "
                            "and all established results. "
                            "Do NOT repeat those. Instead provide: (1) relevant parts of "
                            "the survey background or known methods, (2) strategic context — "
                            "why this task matters and how it fits the plan, (3) relevant "
                            "research notes or observations from prior iterations, "
                            "(4) inter-entity connections the agent should be aware of. "
                            "The agent has NO access to strategy, research notes, or "
                            "survey — relay the relevant parts here."
                        ),
                    },
                    "method_hints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Suggested methods or approaches for the agent to consider."
                        ),
                    },
                    "assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": ("Key assumptions the agent should work under."),
                    },
                    "relevant_results": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "References to established results or prior evidence "
                            "that are relevant to this task (e.g. 'ER-001', 'WH-003')."
                        ),
                    },
                    "recommended_sanity_checks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Sanity checks the agent should verify its result against. "
                            "Pick from the global sanity-checks list or write task-specific "
                            "ones (e.g. 'result must reduce to X in the Y limit')."
                        ),
                    },
                },
                "required": ["target_claim", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_termination",
            "description": (
                "Request termination of the research loop. Use when all RQs "
                "are resolved or abandoned and all WHs are promoted or abandoned. "
                "The system enforces completion gates and reports blockers if not met."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the research is complete.",
                    },
                    "answer_ers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Ordered list of ER IDs that constitute the answer "
                            "(e.g. ['ER-001', 'ER-003', 'ER-005']). The formatter "
                            "uses this to structure the final answer."
                        ),
                    },
                },
                "required": ["answer_ers"],
            },
        },
    },
]
