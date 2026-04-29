"""Tests for sandboxed Python execution."""

import tempfile
from open_dirac.utils.sandbox import execute_python


class TestSoftCheckPattern:
    """Test that the soft-check pattern exits 0 and genuine crashes exit nonzero."""

    def test_soft_check_pattern_exits_zero(self):
        """Script with the new soft-check pattern (some checks fail) exits 0."""
        script = """\
import numpy as np

results = []
test_points = [("a", 1.0, 1.0), ("b", 1.0, 2.0), ("c", 3.0, 3.0)]
for name, lhs, rhs in test_points:
    try:
        ok = np.isclose(lhs, rhs, rtol=1e-6)
        results.append(ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name} -> lhs={lhs}, rhs={rhs}")
    except Exception as e:
        results.append(False)
        print(f"ERROR: {name} -> {e}")
n_passed = sum(results)
n_total = len(results)
print(f"\\nCHECKS: {n_passed}/{n_total} PASSED")
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            result = execute_python(f.name)
        assert result.returncode == 0
        assert "CHECKS: 2/3 PASSED" in result.stdout
        assert "FAIL:" in result.stdout

    def test_genuine_crash_exits_nonzero(self):
        """Script with an ImportError exits with nonzero returncode."""
        script = "import nonexistent_module_xyz\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            result = execute_python(f.name)
        assert result.returncode != 0


def test_execute_simple_script():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('hello world')\n")
        f.flush()
        result = execute_python(f.name)
    assert result.stdout.strip() == "hello world"
    assert result.returncode == 0
    assert not result.timed_out


def test_execute_script_with_error():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("raise ValueError('test error')\n")
        f.flush()
        result = execute_python(f.name)
    assert result.returncode != 0
    assert "ValueError" in result.stderr


def test_execute_timeout():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import time; time.sleep(10)\n")
        f.flush()
        result = execute_python(f.name, timeout=1)
    assert result.timed_out
    assert result.returncode == -1


def test_execute_nonexistent_script():
    result = execute_python("/nonexistent/script.py")
    assert result.returncode != 0


def test_mplbackend_set():
    """Verify sandbox sets MPLBACKEND=Agg in the subprocess environment."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import os; print(os.environ.get('MPLBACKEND', ''))\n")
        f.flush()
        result = execute_python(f.name)
    assert result.returncode == 0
    assert "Agg" in result.stdout


def test_plt_show_no_block():
    """Verify plt.show() does not block with the Agg backend."""
    script = (
        "import matplotlib\n"
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3])\n"
        "plt.show()\n"
        "print('done')\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        result = execute_python(f.name, timeout=60)
    assert not result.timed_out
    assert "done" in result.stdout
