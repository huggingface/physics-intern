"""Tests for the RSA runner (open_dirac.rsa.runner)."""

from open_dirac.rsa.runner import (
    _extract_answer_key,
    _majority_vote,
    build_aggregation_message,
)


# ---------------------------------------------------------------------------
# _extract_answer_key
# ---------------------------------------------------------------------------


class TestExtractAnswerKey:
    def test_final_answer_label(self):
        text = "Some derivation...\n\nFinal Answer: $T_H = 1/(8\\pi M)$"
        key = _extract_answer_key(text)
        assert "$T_H = 1/(8\\pi M)$" in key

    def test_final_answer_bold(self):
        text = "Work...\n\n**Final Answer:** 42"
        key = _extract_answer_key(text)
        assert "42" in key

    def test_code_template_preferred(self):
        """def answer() code block is preferred over Final Answer text."""
        text = (
            "Derivation...\n\n"
            "```python\ndef answer():\n    return 42\n```\n\n"
            "**Final Answer:** 42"
        )
        key = _extract_answer_key(text)
        assert "def answer" in key

    def test_no_answer_returns_empty(self):
        key = _extract_answer_key("Just some reasoning with no conclusion")
        assert key == ""

    def test_empty_string(self):
        assert _extract_answer_key("") == ""

    def test_multiline_final_answer(self):
        text = "Work...\n\nFinal Answer:\n$$E = mc^2$$"
        key = _extract_answer_key(text)
        assert "E = mc^2" in key


# ---------------------------------------------------------------------------
# _majority_vote
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def test_clear_winner(self):
        responses = [
            "Derivation A\n\nFinal Answer: 42",
            "Derivation B\n\nFinal Answer: 42",
            "Derivation C\n\nFinal Answer: 7",
        ]
        winner, count, n_valid = _majority_vote(responses)
        assert count == 2
        assert n_valid == 3
        assert "42" in winner

    def test_single_response(self):
        responses = ["Final Answer: 42"]
        winner, count, n_valid = _majority_vote(responses)
        assert count == 1
        assert n_valid == 1

    def test_no_valid_answers(self):
        responses = ["no answer here", "also nothing"]
        winner, count, n_valid = _majority_vote(responses)
        assert count == 0
        assert n_valid == 0
        assert winner == "no answer here"  # falls back to first

    def test_empty_list(self):
        winner, count, n_valid = _majority_vote([])
        assert winner == ""
        assert count == 0

    def test_tie_uses_first_occurrence(self):
        responses = [
            "Final Answer: A",
            "Final Answer: B",
        ]
        winner, count, n_valid = _majority_vote(responses)
        assert count == 1
        # First occurrence wins
        assert "A" in winner

    def test_all_agree(self):
        responses = [f"Work {i}\n\nFinal Answer: 42" for i in range(5)]
        winner, count, n_valid = _majority_vote(responses)
        assert count == 5
        assert n_valid == 5


# ---------------------------------------------------------------------------
# build_aggregation_message
# ---------------------------------------------------------------------------


class TestBuildAggregationMessage:
    def test_single_candidate(self):
        msg = build_aggregation_message("Problem X", "", ["Solution A"])
        assert "Problem X" in msg
        assert "candidate solution" in msg.lower()
        assert "Solution A" in msg

    def test_multiple_candidates(self):
        msg = build_aggregation_message("Problem X", "", ["Sol A", "Sol B", "Sol C"])
        assert "3 candidate solutions" in msg
        assert "Candidate 1" in msg
        assert "Candidate 2" in msg
        assert "Candidate 3" in msg
        assert "Sol A" in msg
        assert "Sol C" in msg

    def test_with_answer_template(self):
        msg = build_aggregation_message(
            "Problem X", "def answer():\n    pass", ["Sol A"]
        )
        assert "```python" in msg
        assert "def answer():" in msg

    def test_no_template(self):
        msg = build_aggregation_message("Problem X", "", ["Sol A"])
        assert "```python" not in msg
