# OpenDirac

This is a multi-agent scaffolding system for autonomous scientific research in mathematics and theoretical physics.

See README.md for overview, motivation, and design principles.

## Tech Stack

- Use `uv` for dependency management
- Tests: `pytest` (run with `uv run python -m pytest -v`, need `--extra testing`)
- `rich` for console output

## Key Invariants

These are non-obvious rules enforced by the scaffolding. Violating them will break things.

- **`ResearchState` is the single source of truth** — agents render context from it via renderers. No file read-back from disk.
- **MD files are write-only snapshots** — `RESEARCH_STATE.md`, `EVIDENCE_LOG.md`, `CRITIQUE_LOG.md` are rendered once per iteration for git history and `verify.py`. Never read back by agents.
- **Fresh context per call** — every agent gets a new context built from `ResearchState`. No conversation history carries over.
- **Tool definitions use OpenAI canonical format** (`type: "function"`, `function: {name, description, parameters}`). The Anthropic adapter transforms to `input_schema` format.
- **Agent prompts are static `.md` files** co-located with each agent in `src/open_dirac/agents/<name>/prompt.md` — no templating.
- **YAML frontmatter parsing always falls back to regex** — never crash the loop on parse failure.
- **Workspace git is managed by the scaffolding loop**, not by agents.
- **ERs are immutable** — only the adjudicator can demote them (via valid critique). No direct dispatch to ERs.
- **WH→ER promotion is automatic** — `_auto_promote` in `engine.py` fires after VERIFIED review when dependencies are satisfied, with cascading.
- **Iteration counter is scaffolding-maintained** (`_update_research_iteration()`), not LLM-dependent.
- **`BaseAgent.tools` class attribute determines mode** — non-empty → agentic loop (`run_agent_loop`), empty → one-shot (`call_llm`). Both go through the provider abstraction layer.

### Autophysicist-specific

- **Single-agent, stateless iterations** — the Manager gets a fresh context each iteration. Memory is only what was written to PermanentMemory or Scratchpad.
- **PermanentMemory is append-only, Scratchpad is windowed** — permanent memory is fully visible every iteration; scratchpad shows only the last N entries (default: 5).
- **Token budget triggers wind-down** — when per-iteration tokens exceed `--token-budget`, `dispatch_subagent` is removed; at 1.5× budget the iteration is force-terminated.
- **`submit_final_answer` terminates the run** — sets `problem_solved = True` and breaks the iteration loop. Formal evaluation runs automatically after.
