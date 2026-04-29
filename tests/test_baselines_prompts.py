"""Tests for open_dirac.baselines.prompts."""

from open_dirac.baselines.prompts import SYSTEM_PROMPT, build_user_message


class TestSystemPrompt:
    def test_nonempty(self):
        assert SYSTEM_PROMPT.strip()

    def test_mentions_latex(self):
        """Guards against accidental truncation — the prompt must keep
        telling the model to emit LaTeX-delimited math."""
        assert "LaTeX" in SYSTEM_PROMPT


class TestBuildUserMessage:
    def test_plain_problem(self):
        msg = build_user_message("Derive the Hawking temperature.")
        assert msg == "Derive the Hawking temperature."

    def test_strips_whitespace(self):
        msg = build_user_message("  hello  \n")
        assert msg == "hello"

    def test_with_answer_template(self):
        msg = build_user_message("Problem text", "def answer():\n    return 42")
        assert "Problem text" in msg
        assert "```python" in msg
        assert "def answer():" in msg
        assert "Answer template" in msg

    def test_empty_template_ignored(self):
        msg = build_user_message("Problem text", "")
        assert "```python" not in msg
        assert "template" not in msg.lower()
