"""Tests for console output logging and replay."""

from open_dirac.console import LoggingConsole, replay_log


class TestLoggingConsole:
    def test_tee_to_file(self, tmp_path):
        log_path = tmp_path / "console.log"
        c = LoggingConsole()
        c.setup_log(log_path)
        c.print("hello world")
        c._close_log()

        assert "hello world" in log_path.read_text()

    def test_rule_logged(self, tmp_path):
        log_path = tmp_path / "console.log"
        c = LoggingConsole()
        c.setup_log(log_path)
        c.rule("ITERATION 1")
        c._close_log()

        content = log_path.read_text()
        # Rich may insert ANSI codes between words, so check parts separately
        assert "ITERATION" in content
        assert "1" in content

    def test_setup_log_idempotent(self, tmp_path):
        log_path = tmp_path / "console.log"
        c = LoggingConsole()
        c.setup_log(log_path)
        c.setup_log(log_path)  # second call is no-op
        c.print("only once")
        c._close_log()

        assert log_path.read_text().count("only once") == 1

    def test_print_without_setup(self):
        """print() works before setup_log is called."""
        c = LoggingConsole()
        c.print("no log configured")  # should not raise

    def test_setup_creates_parent_dirs(self, tmp_path):
        log_path = tmp_path / "deep" / "nested" / "console.log"
        c = LoggingConsole()
        c.setup_log(log_path)
        c.print("hello")
        c._close_log()

        assert log_path.exists()
        assert "hello" in log_path.read_text()


class TestReplayLog:
    def test_replay_tails_lines(self, tmp_path, capsys):
        log_path = tmp_path / "console.log"
        lines = [f"line {i}" for i in range(100)]
        log_path.write_text("\n".join(lines) + "\n")

        replay_log(log_path, tail=10)
        captured = capsys.readouterr().out
        assert "line 90" in captured
        assert "line 99" in captured
        assert "line 50" not in captured

    def test_replay_all(self, tmp_path, capsys):
        log_path = tmp_path / "console.log"
        log_path.write_text("first\nsecond\nthird\n")

        replay_log(log_path, tail=None)
        captured = capsys.readouterr().out
        assert "first" in captured
        assert "third" in captured

    def test_replay_missing_file(self, tmp_path, capsys):
        replay_log(tmp_path / "nonexistent.log")
        captured = capsys.readouterr().out
        assert captured == ""  # silent no-op
