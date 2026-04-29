"""Permanent memory and scratchpad for the autophysicist Research Manager.

Because each iteration runs with fresh LLM context, these two files are the
Manager's only durable state between iterations:

* :class:`PermanentMemory` — append-only. Every entry ever written is
  injected into every future iteration. Reserved for independently
  verified results.
* :class:`Scratchpad` — append-only on disk, but only the *last N* entries
  (default ``window_size=5``) are injected into the next iteration.
  Storage is full; visibility is windowed — the window is a render-time
  concern, not a retention policy. Older entries still live in
  ``SCRATCHPAD.md`` for git history and post-hoc analysis.

Consequence for the Manager prompt: important intermediate results must be
promoted to permanent memory *before* they scroll off the scratchpad
window, otherwise the next iteration will not see them.
"""

from datetime import datetime, timezone
from pathlib import Path

from ..utils.markdown import tail_entries


class PermanentMemory:
    """Append-only persistent memory. Full contents injected every iteration."""

    FILENAME = "PERMANENT_MEMORY.md"

    def __init__(self, workspace_root: Path):
        self._path = workspace_root / self.FILENAME
        if not self._path.exists():
            self._path.write_text("# Permanent Memory\n\n")

    def append(self, content: str, iteration: int) -> str:
        """Append an iteration-tagged entry. Return confirmation message."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        entry = f"\n## Iteration {iteration} — {timestamp}\n\n{content.strip()}\n"
        with open(self._path, "a") as f:
            f.write(entry)
        return f"Written to permanent memory (iteration {iteration}, {len(content)} chars)."

    def read_full(self) -> str:
        """Read the entire permanent memory file."""
        return self._path.read_text() if self._path.exists() else ""

    @property
    def size_chars(self) -> int:
        return len(self.read_full())


class Scratchpad:
    """Rolling append-only scratchpad. Only last N entries injected."""

    FILENAME = "SCRATCHPAD.md"

    def __init__(self, workspace_root: Path, window_size: int = 5):
        self._path = workspace_root / self.FILENAME
        self._window_size = window_size
        self._entry_count = 0
        if not self._path.exists():
            self._path.write_text("# Scratchpad\n\n")
        else:
            # Count existing entries for resume
            text = self._path.read_text()
            self._entry_count = text.count("\n## Iteration ")

    def append(self, content: str, iteration: int) -> str:
        """Append an iteration-tagged entry. Return confirmation message."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        entry = f"\n## Iteration {iteration} — {timestamp}\n\n{content.strip()}\n"
        with open(self._path, "a") as f:
            f.write(entry)
        self._entry_count += 1
        visible = min(self._entry_count, self._window_size)
        return (
            f"Written to scratchpad (iteration {iteration}, {len(content)} chars). "
            f"{visible} of {self._entry_count} entries visible next iteration."
        )

    def read_window(self) -> str:
        """Read the last N entries for context injection."""
        text = self._path.read_text() if self._path.exists() else ""
        if not text.strip():
            return ""
        return tail_entries(text, self._window_size)

    def read_full(self) -> str:
        """Read entire scratchpad (for logging/debug)."""
        return self._path.read_text() if self._path.exists() else ""

    @property
    def entry_count(self) -> int:
        return self._entry_count
